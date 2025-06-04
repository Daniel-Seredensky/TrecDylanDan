package src.InfoRetrieval;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.highlight.Highlighter;
import org.apache.lucene.search.highlight.QueryScorer;
import org.apache.lucene.search.highlight.SimpleHTMLFormatter;
import org.apache.lucene.search.highlight.SimpleFragmenter;
import org.apache.lucene.search.highlight.TokenSources;
import org.apache.lucene.search.highlight.InvalidTokenOffsetsException;
import org.apache.lucene.index.Terms;
import org.apache.lucene.analysis.TokenStream;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

public class Search {
    private static final String INDEX_DIR = "DerivedData/CluewebIndex";
    private static final int TOP_N_PER_QUERY = 300;
    private static final int FINAL_TOP_N = 150;
    private static final int MAX_QUERIES = 12;
    
    // Highlighting parameters
    private static final int FRAGMENT_SIZE = 150; // Characters per fragment
    private static final int MAX_FRAGMENTS = 3;   // Will help us get ~6 sentences
    private static final String FRAGMENT_SEPARATOR = " ... ";
    
    // RRF parameter (commonly used value)
    private static final double RRF_K = 60.0;

    // Class to hold search results with RRF scoring
    private static class SearchResult {
        public final String url;
        public final String id;
        public final String content;
        public final String highlightedContent;
        public double rrfScore;
        public final Map<String, Integer> queryRanks; // Track rank for each query
        public final Map<String, Float> queryScores;  // Track original BM25 scores

        public SearchResult(String url, String id, String content, String highlightedContent, 
                          String queryId, int rank, float bm25Score) {
            this.url = url;
            this.id = id;
            this.content = content;
            this.highlightedContent = highlightedContent;
            this.rrfScore = 0.0;
            this.queryRanks = new HashMap<>();
            this.queryScores = new HashMap<>();
            
            this.queryRanks.put(queryId, rank);
            this.queryScores.put(queryId, bm25Score);
        }

        // Add ranking from another query for RRF calculation
        public void addQueryResult(String queryId, int rank, float bm25Score) {
            this.queryRanks.put(queryId, rank);
            this.queryScores.put(queryId, bm25Score);
        }
        
        // Calculate RRF score based on all query rankings
        public void calculateRRFScore() {
            this.rrfScore = queryRanks.values().stream()
                    .mapToDouble(rank -> 1.0 / (RRF_K + rank))
                    .sum();
        }
        
        public int getQueryCount() {
            return queryRanks.size();
        }
    }

    // Class to hold individual query results before merging
    private static class QueryResult {
        public final String queryId;
        public final List<RankedDocument> documents;
        
        public QueryResult(String queryId, List<RankedDocument> documents) {
            this.queryId = queryId;
            this.documents = documents;
        }
    }
    
    private static class RankedDocument {
        public final String url;
        public final String id;
        public final String content;
        public final String highlightedContent;
        public final float bm25Score;
        public final int rank;
        
        public RankedDocument(String url, String id, String content, String highlightedContent, 
                            float bm25Score, int rank) {
            this.url = url;
            this.id = id;
            this.content = content;
            this.highlightedContent = highlightedContent;
            this.bm25Score = bm25Score;
            this.rank = rank;
        }
    }

    public static void main(String[] args) {
        if (args.length < 2 || args.length > MAX_QUERIES + 1) {
            System.out.println("Provided " + args.length + " arguments");
            System.err.println("Usage: java Search <query1> [query2] ... [query12] <result_path>");
            System.err.println("Maximum 12 queries supported, plus result path");
            System.exit(1);
        }
        
        // Pop the last element as RESULT_PATH
        String RESULT_PATH = args[args.length - 1];
        
        // Create queries list from all elements except the last one
        List<String> queries = Arrays.asList(Arrays.copyOf(args, args.length - 1));
        
        try {
            // Open index
            Directory dir = NIOFSDirectory.open(Paths.get(INDEX_DIR));
            DirectoryReader reader = DirectoryReader.open(dir);
            
            // Execute parallel searches with improved multi-query handling
            List<SearchResult> mergedResults = executeParallelSearchesWithRRF(reader, queries);
            
            // Write results to file
            writeResults(mergedResults, RESULT_PATH);
            
            reader.close();
            dir.close();
            
            System.out.println("Search completed with " + queries.size() + " queries.");
            System.out.println("Found " + mergedResults.size() + " unique results.");
            
            // Print RRF statistics
            printRRFStatistics(mergedResults);
            System.out.println("Results written to: " + RESULT_PATH);
            
        } catch (Exception e) {
            System.err.println("Error executing search: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static List<SearchResult> executeParallelSearchesWithRRF(DirectoryReader reader, List<String> queries) 
            throws InterruptedException, ExecutionException {
        
        ExecutorService executor = Executors.newFixedThreadPool(Math.min(queries.size(), 4));
        List<Future<QueryResult>> futures = new ArrayList<>();

        // Submit search tasks - each returns ranked results for one query
        for (int i = 0; i < queries.size(); i++) {
            final String queryStr = queries.get(i);
            final String queryId = "Q" + (i + 1);
            
            Future<QueryResult> future = executor.submit(() -> {
                try {
                    return executeSearchWithHighlighting(reader, queryStr, queryId);
                } catch (Exception e) {
                    System.err.println("Error searching for query '" + queryStr + "': " + e.getMessage());
                    e.printStackTrace();
                    return new QueryResult(queryId, new ArrayList<>());
                }
            });
            futures.add(future);
        }

        // Collect all query results
        List<QueryResult> allQueryResults = new ArrayList<>();
        for (Future<QueryResult> future : futures) {
            QueryResult queryResult = future.get();
            allQueryResults.add(queryResult);
        }

        executor.shutdown();

        // Apply RRF fusion
        return applyRRFFusion(allQueryResults);
    }

    private static QueryResult executeSearchWithHighlighting(DirectoryReader reader, String queryStr, String queryId) 
            throws Exception {
        
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity());

        // Prepare analyzer and parser
        StandardAnalyzer analyzer = new StandardAnalyzer();
        QueryParser parser = new QueryParser("content", analyzer);
        org.apache.lucene.search.Query query = parser.parse(queryStr);

        // Execute search
        TopDocs topDocs = searcher.search(query, TOP_N_PER_QUERY);
        ScoreDoc[] hits = topDocs.scoreDocs;

        // Setup highlighter
        QueryScorer scorer = new QueryScorer(query);
        SimpleHTMLFormatter formatter = new SimpleHTMLFormatter("", "");
        Highlighter highlighter = new Highlighter(formatter, scorer);
        highlighter.setTextFragmenter(new SimpleFragmenter(FRAGMENT_SIZE));

        List<RankedDocument> documents = new ArrayList<>();
        
        for (int i = 0; i < hits.length; i++) {
            ScoreDoc sd = hits[i];
            Document doc = searcher.doc(sd.doc);
            String url = doc.get("url");
            String id = doc.get("id");
            String content = doc.get("content");
            
            if (content != null) {
                String highlightedContent = getHighlightedFragment(
                    highlighter, analyzer, content, reader, sd.doc);
                
                documents.add(new RankedDocument(url, id, content, highlightedContent, 
                                               sd.score, i + 1)); // rank is 1-based
            }
        }

        return new QueryResult(queryId, documents);
    }

    private static String getHighlightedFragment(Highlighter highlighter, StandardAnalyzer analyzer, 
                                               String content, DirectoryReader reader, int docId) {
        try {
            // Use analyzer to create token stream directly
            TokenStream tokenStream = analyzer.tokenStream("content", content);
            
            String[] fragments = highlighter.getBestFragments(tokenStream, content, MAX_FRAGMENTS);
            
            if (fragments.length > 0) {
                return String.join(FRAGMENT_SEPARATOR, fragments);
            } else {
                // Fallback to simple truncation if highlighting fails
                return content.length() > (FRAGMENT_SIZE * MAX_FRAGMENTS) 
                    ? content.substring(0, FRAGMENT_SIZE * MAX_FRAGMENTS) + "..."
                    : content;
            }
            
        } catch (IOException | InvalidTokenOffsetsException e) {
            System.err.println("Error highlighting content: " + e.getMessage());
            // Fallback to simple truncation
            return content.length() > (FRAGMENT_SIZE * MAX_FRAGMENTS) 
                ? content.substring(0, FRAGMENT_SIZE * MAX_FRAGMENTS) + "..."
                : content;
        }
    }

    private static List<SearchResult> applyRRFFusion(List<QueryResult> allQueryResults) {
        Map<String, SearchResult> resultMap = new HashMap<>();
        
        // Process each query's results
        for (QueryResult queryResult : allQueryResults) {
            for (RankedDocument doc : queryResult.documents) {
                String key = doc.id != null ? doc.id : doc.url;
                if (key != null) {
                    SearchResult existing = resultMap.get(key);
                    if (existing == null) {
                        // First time seeing this document
                        SearchResult newResult = new SearchResult(
                            doc.url, doc.id, doc.content, doc.highlightedContent,
                            queryResult.queryId, doc.rank, doc.bm25Score);
                        resultMap.put(key, newResult);
                    } else {
                        // Document found by multiple queries
                        existing.addQueryResult(queryResult.queryId, doc.rank, doc.bm25Score);
                    }
                }
            }
        }
        
        // Calculate RRF scores for all documents
        for (SearchResult result : resultMap.values()) {
            result.calculateRRFScore();
        }
        
        // Sort by RRF score descending and take top N
        return resultMap.values().stream()
                .sorted((a, b) -> Double.compare(b.rrfScore, a.rrfScore))
                .limit(FINAL_TOP_N)
                .collect(Collectors.toList());
    }

    private static void printRRFStatistics(List<SearchResult> results) {
        long multiQueryDocs = results.stream()
                .mapToInt(SearchResult::getQueryCount)
                .filter(count -> count > 1)
                .count();
        
        double avgRRFScore = results.stream()
                .mapToDouble(r -> r.rrfScore)
                .average()
                .orElse(0.0);
        
        int maxQueryCount = results.stream()
                .mapToInt(SearchResult::getQueryCount)
                .max()
                .orElse(0);
        
        System.out.println("Documents found by multiple queries: " + multiQueryDocs);
        System.out.println("Average RRF score: " + String.format("%.4f", avgRRFScore));
        System.out.println("Maximum queries per document: " + maxQueryCount);
    }

    private static void writeResults(List<SearchResult> results,String RESULT_PATH) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(RESULT_PATH))) {
            for (SearchResult result : results) {
                ObjectNode node = mapper.createObjectNode();
                node.put("url", result.url);
                node.put("id", result.id);
                node.put("bestContent", result.highlightedContent);
                node.put("rrf_score", result.rrfScore);
                node.put("query_count", result.getQueryCount());
                
                // Add query-specific information
                ObjectNode queryInfo = mapper.createObjectNode();
                for (Map.Entry<String, Integer> entry : result.queryRanks.entrySet()) {
                    ObjectNode qInfo = mapper.createObjectNode();
                    qInfo.put("rank", entry.getValue());
                    qInfo.put("bm25_score", result.queryScores.get(entry.getKey()));
                    queryInfo.set(entry.getKey(), qInfo);
                }
                node.set("query_details", queryInfo);

                writer.write(mapper.writeValueAsString(node));
                writer.newLine();
            }
        }
    }
}
