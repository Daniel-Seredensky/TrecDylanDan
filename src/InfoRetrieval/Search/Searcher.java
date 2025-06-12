package src.InfoRetrieval.Search;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

public class Searcher {
    private static final String INDEX_DIR       = "DerivedData/MSMarcoIndex";
    private static final int    TOP_N_PER_QUERY = 1500;
    private static final int    FINAL_TOP_N     = 600;
    private static final int    MAX_QUERIES     = 4;
    private static final double RRF_K           = 60.0;

    // single shared analyzer with synonyms, lowercase, stop, and stem
    private static final Analyzer QUERY_ANALYZER;
    static {
        try {
            Map<String,String> synArgs = new HashMap<>();
            synArgs.put("synonyms",   "synonyms.txt");
            synArgs.put("format",     "solr");
            synArgs.put("ignoreCase", "true");
            synArgs.put("expand",     "true");

            Map<String,String> stopArgs = new HashMap<>();
            stopArgs.put("ignoreCase", "true");

            QUERY_ANALYZER = CustomAnalyzer.builder(Paths.get("src/InfoRetrieval/Search/synonyms"))
                .withTokenizer("standard")
                .addTokenFilter("englishPossessive")
                .addTokenFilter("synonymGraph", synArgs)
                .addTokenFilter("lowercase")
                .addTokenFilter("stop", stopArgs)
                .addTokenFilter("porterStem")
                .build();

        } catch (IOException e) {
            throw new RuntimeException("Unable to create analyzer", e);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > MAX_QUERIES + 1) {
            System.err.println("Usage: java Searcher <q1>...[q4] <result_path>");
            System.exit(1);
        }

        String resultPath = args[args.length - 1];
        List<String> queries = List.of(args).subList(0, args.length - 1);

        try (Directory dir = NIOFSDirectory.open(Paths.get(INDEX_DIR));
             DirectoryReader reader = DirectoryReader.open(dir)) {

            List<SearchResult> results = executeParallelSearchesWithRRF(reader, queries);
            writeResults(results, resultPath);

            System.out.printf("Search: %d queries; %d unique docs (collapsed).%n",
                              queries.size(), results.size());
            printRRFStatistics(results);
        }
    }

    private static List<SearchResult> executeParallelSearchesWithRRF(DirectoryReader reader,
                                                                    List<String> queries)
      throws InterruptedException, ExecutionException {

        ExecutorService exec = Executors.newFixedThreadPool(Math.min(queries.size(), 4));
        List<Future<QueryResult>> futures = new ArrayList<>();

        for (int i = 0; i < queries.size(); i++) {
            final String q   = queries.get(i);
            final String qid = "Q" + (i + 1);
            futures.add(exec.submit(() -> searchShard(reader, q, qid)));
        }
        exec.shutdown();

        List<QueryResult> all = new ArrayList<>();
        for (Future<QueryResult> f : futures) {
            all.add(f.get());
        }
        return applyRRFFusion(all);
    }

    private static QueryResult searchShard(DirectoryReader reader,
                                           String queryStr,
                                           String queryId)
      throws Exception {

        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity());
        StoredFields stored = searcher.storedFields();

        Map<String, Float> boosts = Map.of(
            "title",    2.0f,
            "headings", 1.3f,
            "segment",  1.0f
        );
        String[] fields = {"title","headings","segment"};
        MultiFieldQueryParser parser =
            new MultiFieldQueryParser(fields, QUERY_ANALYZER, boosts);
        Query query = parser.parse(queryStr);

        TopDocs td = searcher.search(query, TOP_N_PER_QUERY);
        List<RankedDocument> docs = new ArrayList<>();

        for (int i = 0; i < td.scoreDocs.length; i++) {
            ScoreDoc sd      = td.scoreDocs[i];
            Document d       = stored.document(sd.doc);
            String segment   = d.get("segment");
            String segmentId = d.get("segment_id");
            String url       = d.get("url");
            String title     = d.get("title");
            String headings  = d.get("headings");

            // no highlighting—just return the full segment
            docs.add(new RankedDocument(
                url,
                segmentId,
                segment,
                title,
                headings,
                "",         // Old from testing
                sd.score,
                i + 1
            ));
        }
        return new QueryResult(queryId, docs);
    }

    private static List<SearchResult> applyRRFFusion(List<QueryResult> allQR) {
        Map<String, SearchResult> map = new HashMap<>();

        for (QueryResult qr : allQR) {
            for (RankedDocument rd : qr.getDocuments()) {
                String docKey = rd.getFullDocId();
                String fullId = rd.getSegmentId();
                SearchResult sr = map.get(docKey);
                if (sr == null) {
                    sr = new SearchResult(
                        rd.getUrl(),
                        docKey,
                        fullId,
                        rd.getContent(),
                        rd.getTitle(),
                        rd.getHeadings(),
                        rd.getHighlightedContent(),
                        qr.getQueryId(),
                        rd.getRank(),
                        rd.getBm25Score()
                    );
                    map.put(docKey, sr);
                } else {
                    sr.addQueryResult(qr.getQueryId(), rd.getRank(), rd.getBm25Score());
                }
            }
        }

        map.values().forEach(r -> r.calculateRrfScore(RRF_K));
        return map.values().stream()
                  .sorted(Comparator.comparingDouble(SearchResult::getRrfScore).reversed())
                  .limit(FINAL_TOP_N)
                  .collect(Collectors.toList());
    }

    private static void printRRFStatistics(List<SearchResult> res) {
        long multi = res.stream().filter(r -> r.getQueryCount() > 1).count();
        double avg  = res.stream().mapToDouble(SearchResult::getRrfScore).average().orElse(0);
        int max     = res.stream().mapToInt(SearchResult::getQueryCount).max().orElse(0);
        System.out.printf("Docs by >1 query: %d, Avg RRF: %.4f, Max per doc: %d%n",
                          multi, avg, max);
    }

    private static void writeResults(List<SearchResult> res, String out) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(out))) {
            for (SearchResult r : res) {
                ObjectNode n = mapper.createObjectNode();
                n.put("url",         r.getUrl());
                n.put("id",          r.getId());    // full doc id
                n.put("segment",     r.getContent());      // full segment text
                n.put("rrf_score",   r.getRrfScore());
                n.put("query_count", r.getQueryCount());
                n.put("segmentId",   r.getSegmentId());
                n.put("title",       r.getTitle());
                n.put("headings",    r.getHeadings());

                ObjectNode detail = mapper.createObjectNode();
                r.getQueryRanks().forEach((qid, rank) -> {
                    ObjectNode qi = mapper.createObjectNode();
                    qi.put("rank",      rank);
                    qi.put("bm25_score", r.getQueryScores().get(qid));
                    detail.set(qid, qi);
                });
                n.set("query_details", detail);

                writer.write(mapper.writeValueAsString(n));
                writer.newLine();
            }
        }
    }
}
