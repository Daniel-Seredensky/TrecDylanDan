/*  -----------------------------------------------------------------------------
// Searcher.java — concurrency + synonym map + RRF fusion + raw JSON output
//                 *collapsed on full‑doc id*
// -----------------------------------------------------------------------------
*/

package src.IR_Ensemble.QA_Assistant.Search;

import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.CompletableFuture;

public class Searcher {

    // ---------------- configuration ----------------
    private static final int    TOP_N_PER_QUERY = 1000;
    private static final int    FINAL_N         = 200;          
    private static final double RRF_K           = 60.0;
    private static final int    MAX_QUERIES     = 8;

    private static final com.fasterxml.jackson.databind.ObjectMapper MAPPER =
        new ObjectMapper();

    // single shared analyzer with synonyms, lowercase, stop, and stem
    private static final CustomAnalyzer QUERY_ANALYZER;
    static {
        try {
            Map<String,String> synArgs = new HashMap<>();
            synArgs.put("synonyms",   "synonyms.txt");
            synArgs.put("format",     "solr");
            synArgs.put("ignoreCase", "true");
            synArgs.put("expand",     "true");

            Map<String,String> stopArgs = new HashMap<>();
            stopArgs.put("ignoreCase", "true");

            QUERY_ANALYZER = 
                CustomAnalyzer.builder(Paths.get("src/IR_Ensemble/QA_Assistant/Search/synonyms"))
                    .withTokenizer("standard")                  // StandardTokenizer
                    .addTokenFilter("englishPossessive")        // 's →   (keeps positions)
                    .addTokenFilter("lowercase")
                    .addTokenFilter("stop", stopArgs)           // same default EN stop set
                    .addTokenFilter("synonymGraph", synArgs)    // Synonym expansion
                    .addTokenFilter("porterStem")
                    .build();

        } catch (IOException e) {
            throw new RuntimeException("Unable to create analyzer", e);
        }
    }

    /* ---------------- per‑rootId aggregate ---------------- */
    private static class Aggregate {
        double score;            // cumulative RRF score across queries
        double bestSegBoost;     // per‑query boost of best segment so far
        String bestRaw;          // raw JSON for that segment
    }

    // ---------------- main ----------------
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java Searcher <query1> [<query2> ... <query8>] <output.jsonl>");
            System.exit(1);
        }
        if (args.length - 1 > MAX_QUERIES) {
            System.err.println("Supports up to " + MAX_QUERIES + " queries.");
            System.exit(1);
        }

        String outPath = args[args.length - 1];
        List<String> queries = Arrays.asList(Arrays.copyOfRange(args, 0, args.length - 1));

        DirectoryReader reader = LuceneHolder.getReader();
        IndexSearcher   searcher = LuceneHolder.getSearcher();

        /* one Aggregate per *full doc id* */
        ConcurrentHashMap<String, Aggregate> aggMap = new ConcurrentHashMap<>();

        ExecutorService pool    = LuceneHolder.getLucenePool();
        List<CompletableFuture<Void>> cf = new ArrayList<>();

        for (int i = 0; i < queries.size(); i++) {
            final String qtext = queries.get(i);
            cf.add(CompletableFuture.runAsync(
                    () -> runSingleQuery(qtext, searcher, aggMap), pool));
        }

        CompletableFuture
                .allOf(cf.toArray(new CompletableFuture[0]))
                .join();

        /* ---------------- sort & write top N ---------------- */
        List<Aggregate> sorted = new ArrayList<>(aggMap.values());
        sorted.sort((a, b) -> Double.compare(b.score, a.score));

        try (BufferedWriter w = Files.newBufferedWriter(Paths.get(outPath))) {
            int count = 0;
            for (Aggregate ag : sorted) {
                String oneLine = MAPPER.writeValueAsString(
                                     MAPPER.readTree(ag.bestRaw));   // compact form
                w.write(oneLine);
                w.newLine();
                if (++count == FINAL_N) break;
            }
        }
    }

    /* ---------------- helper: run single query ---------------- */
    private static void runSingleQuery(String qtext,
                                       IndexSearcher searcher,
                                       ConcurrentHashMap<String, Aggregate> aggMap) {
        try {
            Query query = new QueryParser("contents", QUERY_ANALYZER).parse(qtext);
            TopDocs td  = searcher.search(query, TOP_N_PER_QUERY);

            /* ensure only the *highest‑ranked* segment per rootId
               influences this query's scoring */
            HashSet<String> seenRootIds = new HashSet<>();

            for (int rank = 0; rank < td.scoreDocs.length; rank++) {
                ScoreDoc sd  = td.scoreDocs[rank];
                Document doc = searcher.storedFields().document(sd.doc);
                String segId = doc.get("id");
                int hashPos  = segId.indexOf('#');
                if (hashPos <= 0) continue;                    // safety check

                String rootId = segId.substring(0, hashPos);
                if (!seenRootIds.add(rootId)) continue;        // already handled

                double rrfBoost = 1.0 / (RRF_K + (rank + 1));
                String raw      = doc.get("raw");

                /* atomically merge/update the aggregate */
                aggMap.compute(rootId, (k, ag) -> {
                    if (ag == null) ag = new Aggregate();
                    ag.score += rrfBoost;
                    if (rrfBoost > ag.bestSegBoost) {
                        ag.bestSegBoost = rrfBoost;
                        ag.bestRaw      = raw;
                    }
                    return ag;
                });
            }
        } catch (Exception e) {
            throw new RuntimeException("Query failed: \"" + qtext + "\"", e);
        }
    }
}
