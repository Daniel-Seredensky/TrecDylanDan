package src.QA_Assistant.Search;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.StringHelper;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class DocumentSelection {

    // ----------------- config -----------------
    private static final Path INDEX_PATH   = Paths.get("/Volumes/X9 Pro/MarcoIndex");
    private static final int  MAX_SEG_IDS  = 4;
    private static final ObjectMapper JSON = new ObjectMapper();

    // ----------------- main -------------------
    public static void main(String[] args) throws Exception {

        boolean asSegments = false;
        List<String> ids   = new ArrayList<>();

        // --- parse CLI flags / ids ------------
        for (String arg : args) {
            switch (arg) {
                case "--asSegments" -> asSegments = true;
                default             -> ids.add(arg);
            }
        }

        // --- validate -------------------------
        if (ids.isEmpty()) {
            System.err.println("No document IDs supplied.");
            System.exit(1);
        }
        if (asSegments && ids.size() > MAX_SEG_IDS) {
            System.err.println("With --asSegments you may pass up to " + MAX_SEG_IDS + " IDs.");
            System.exit(1);
        }
        if (!asSegments && ids.size() > 1) {
            // Full‑text mode: keep only the first ID
            ids = List.of(ids.get(0));
        }

        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(INDEX_PATH))) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            if (asSegments) {
                for (String id : ids) {
                    ObjectNode n = fetchSegment(id, searcher);
                    System.out.println(JSON.writeValueAsString(n));
                }
            } else {
                ObjectNode n = fetchFull(ids.get(0), reader);
                System.out.println(JSON.writeValueAsString(n));
            }
        }
    }

    // -------- exact‑segment retrieval --------
    private static ObjectNode fetchSegment(String segId, IndexSearcher s) throws Exception {
        Query   q   = new TermQuery(new Term("id", segId));
        TopDocs td  = s.search(q, 1);
        if (td.totalHits.value() == 0) {
            throw new IllegalArgumentException("ID not found: " + segId);
        }
        Document  d   = s.storedFields().document(td.scoreDocs[0].doc);
        JsonNode  raw = JSON.readTree(d.get("raw"));

        ObjectNode out = JSON.createObjectNode();
        out.put("id", segId);
        out.put("segment", raw.path("segment").asText(""));
        return out;
    }

    // -------- full‑document assembly ---------
    private static ObjectNode fetchFull(String segId, DirectoryReader reader) throws Exception {
        String stem = segId.contains("#")
                      ? segId.substring(0, segId.indexOf('#') + 1)
                      : segId + "#";  // ensure trailing #

        List<JsonNode> segs = new ArrayList<>();

        TermsEnum te = reader.leaves().get(0).reader().terms("id").iterator();
        if (te.seekCeil(new BytesRef(stem)) == TermsEnum.SeekStatus.END) {
            throw new IllegalArgumentException("No segments found for " + segId);
        }

        do {
            BytesRef term = te.term();
            if (!StringHelper.startsWith(term, new BytesRef(stem))) break;  // prefix exhausted

            PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
            while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                Document d = reader.storedFields().document(pe.docID());
                segs.add(JSON.readTree(d.get("raw")));
            }
        } while (te.next() != null);

        segs.sort(Comparator.comparingInt(n -> n.path("start_char").asInt(0)));

        StringBuilder full = new StringBuilder();
        int currentEnd = -1;  // last character already copied (global coordinate)

        for (JsonNode n : segs) {
            String text = n.path("segment").asText("");
            int start   = n.path("start_char").asInt(0);
            int end     = n.path("end_char").asInt(text.length() + start);

            if (end <= currentEnd) continue;          // this segment is wholly inside material we’ve kept

            int beginInSeg = Math.max(0, currentEnd - start); // skip overlap inside this segment
            if (beginInSeg < text.length()) {
                full.append(text.substring(beginInSeg)).append(' ');
                currentEnd = end;                     // advance frontier
            }
        }
        ObjectNode out = JSON.createObjectNode();
        out.put("id",segId);
        out.put("fullText", full.toString());
        return out;
    }
}
