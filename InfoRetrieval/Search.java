package InfoRetrieval;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

public class Search {
    private static final String INDEX_DIR = "DerivedData/CluewebIndex";
    private static final int TOP_N = 20;
    private static final int CONTENT_PREVIEW_LENGTH = 200;

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java Search <query> <output-jsonl-path>");
            System.exit(1);
        }

        // Last arg is output file, rest form the query
        String outputPath = "DerivedData/SearchResults/" + args[args.length - 1];
        String queryStr = String.join(" ", Arrays.copyOfRange(args, 0, args.length - 1));

        try {
            // Open index
            Directory dir = NIOFSDirectory.open(Paths.get(INDEX_DIR));
            DirectoryReader reader = DirectoryReader.open(dir);
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            // Prepare analyzer and parser
            StandardAnalyzer analyzer = new StandardAnalyzer();
            QueryParser parser = new QueryParser("content", analyzer);
            org.apache.lucene.search.Query query = parser.parse(queryStr);

            // Execute search
            TopDocs topDocs = searcher.search(query, TOP_N);
            ScoreDoc[] hits = topDocs.scoreDocs;

            // Setup JSON writer
            ObjectMapper mapper = new ObjectMapper();
            try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputPath))) {
                for (ScoreDoc sd : hits) {
                    Document doc = searcher.doc(sd.doc);
                    String url = doc.get("url");
                    String id = doc.get("id");
                    String content = doc.get("content");
                    String preview = (content.length() > CONTENT_PREVIEW_LENGTH)
                                     ? content.substring(0, CONTENT_PREVIEW_LENGTH)
                                     : content;

                    ObjectNode node = mapper.createObjectNode();
                    node.put("url", url);
                    node.put("id", id);
                    node.put("content", preview);

                    writer.write(mapper.writeValueAsString(node));
                    writer.newLine();
                }
            }

            reader.close();
            dir.close();
        } catch (Exception e) {
            System.err.println("Error executing search: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

