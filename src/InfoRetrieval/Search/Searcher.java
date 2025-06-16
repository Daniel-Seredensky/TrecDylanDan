/*  -----------------------------------------------------------------------------
// Searcher.java — concurrency + synonym map + RRF fusion + raw
//                              JSON output 
// -----------------------------------------------------------------------------
// Package: src.InfoRetrieval.Search
//
//  • Up to 8 LLM‑generated queries run concurrently.
//  • Analyzer includes synonymGraph + Porter stemming (unchanged).
//  • Single‑pass BM25 search on the "contents" field per query.
//  • RRF fusion (k = 60) across all queries.
//  • Output: top 600 documents’ raw JSON lines to <output>.jsonl.
//
//  Dependencies: Lucene 10.x.
// -----------------------------------------------------------------------------
*/
package src.InfoRetrieval.Search;

import org.apache.lucene.analysis.Analyzer;
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

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Searcher {
    // ---------------- configuration ----------------
    private static final Path   INDEX_PATH      = Paths.get("./MarcoIndex");
    private static final int    TOP_N_PER_QUERY = 5000;
    private static final int    FINAL_N         = 750;
    private static final double RRF_K           = 60.0;
    private static final int    MAX_QUERIES     = 8;   

    private static final com.fasterxml.jackson.databind.ObjectMapper MAPPER =
        new com.fasterxml.jackson.databind.ObjectMapper();

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
                CustomAnalyzer.builder(Paths.get("src/InfoRetrieval/Search/synonyms"))
                    .withTokenizer("standard")                  // StandardTokenizer
                    .addTokenFilter("englishPossessive")        // ’s →   (keeps positions)
                    .addTokenFilter("lowercase")
                    .addTokenFilter("stop", stopArgs)           // same default EN stop set
                    .addTokenFilter("synonymGraph", synArgs)    // Synonym expansion
                    .addTokenFilter("porterStem")
                    .build();

        } catch (IOException e) {
            throw new RuntimeException("Unable to create analyzer", e);
        }
    }

    // ---------------- main ----------------
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java ConcurrentRRFSearcher <query1> [<query2> ... <query8>] <output.jsonl>");
            System.exit(1);
        }
        if (args.length - 1 > MAX_QUERIES) {
            System.err.println("Supports up to " + MAX_QUERIES + " queries.");
            System.exit(1);
        }

        String outPath = args[args.length - 1];
        List<String> queries = Arrays.asList(Arrays.copyOfRange(args, 0, args.length - 1));

        DirectoryReader reader = DirectoryReader.open(FSDirectory.open(INDEX_PATH));
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity());

        ConcurrentHashMap<String, Double> scoreMap = new ConcurrentHashMap<>();
        ConcurrentHashMap<String, String> rawMap   = new ConcurrentHashMap<>();

        ExecutorService pool = Executors.newFixedThreadPool(queries.size());
        List<Future<?>> futures = new ArrayList<>();

        for (int i = 0; i < queries.size(); i++) {
            final String qid   = "Q" + (i + 1);
            final String qtext = queries.get(i);
            futures.add(pool.submit(() -> runSingleQuery(qid, qtext, searcher, scoreMap, rawMap)));
        }
        pool.shutdown();
        for (Future<?> f : futures) f.get();
        reader.close();

        List<Map.Entry<String, Double>> sorted = new ArrayList<>(scoreMap.entrySet());
        sorted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        try (BufferedWriter w = Files.newBufferedWriter(Paths.get(outPath))) {
            int count = 0;
            for (Map.Entry<String, Double> e : sorted) {
                String raw = rawMap.get(e.getKey());          // may contain embedded new‑lines
                String oneLine = MAPPER.writeValueAsString(
                                    MAPPER.readTree(raw));    // parse → emit compact form
                w.write(oneLine);
                w.newLine();
                if (++count == FINAL_N) break;
            }
        }
    }

    // ---------------- helper: run single query ----------------
    private static void runSingleQuery(String qid,
                                       String qtext,
                                       IndexSearcher searcher,
                                       ConcurrentHashMap<String, Double> scoreMap,
                                       ConcurrentHashMap<String, String> rawMap) {
        try {
            Query query = new QueryParser("contents", QUERY_ANALYZER).parse(qtext);
            TopDocs td = searcher.search(query, TOP_N_PER_QUERY);

            for (int rank = 0; rank < td.scoreDocs.length; rank++) {
                ScoreDoc sd  = td.scoreDocs[rank];
                Document doc = searcher.storedFields().document(sd.doc);
                String docId = doc.get("id");
                String raw   = doc.get("raw");
                double add   = 1.0 / (RRF_K + (rank + 1));
                scoreMap.merge(docId, add, Double::sum);
                rawMap.putIfAbsent(docId, raw);
            }
        } catch (Exception e) {
            throw new RuntimeException("Query " + qid + " failed", e);
        }
    }
}
