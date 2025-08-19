package src.IR_Ensemble.QA_Assistant.Search;

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
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.StringHelper;

import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Iterate across all leaves, or call reader.termVectors() if you store vectors.
 * Use IndexReader.termDocsEnum(new Term("id", stem)) for direct seek.
 */


/**
 * Revised to parallel‑process segment or document retrieval using the shared
 * {@link ExecutorService} from {@code LuceneHolder.getPool()}.
 */
public final class DocumentSelection {

    // ----------------- config -----------------
    private static final int MAX_SEG_IDS = 6;
    private static final ObjectMapper JSON = new ObjectMapper();

    // ----------------- main -------------------
    public static String run(List<String> args) throws Exception {

        boolean asSegments = args.remove("--asSegments");
        if (args.isEmpty())   throw new IllegalArgumentException("no ids supplied");
        if (asSegments && args.size() > MAX_SEG_IDS)
            throw new IllegalArgumentException("max " + MAX_SEG_IDS + " ids with --asSegments");
        if (!asSegments && args.size() > 1)                   // full‑doc mode
            args = List.of(args.get(0));

        DirectoryReader reader   = LuceneHolder.getReader();
        IndexSearcher   searcher = LuceneHolder.getSearcher();
        searcher.setSimilarity(new BM25Similarity());

        ExecutorService pool = LuceneHolder.getLucenePool();
        List<Callable<JsonNode>> tasks = new ArrayList<>();

        if (asSegments) {
            for (String id : args)
                tasks.add(() -> fetchSegment(id, searcher));
        } else {
            String id = args.get(0);
            tasks.add(() -> fetchFull(id, reader));
        }

        List<CompletableFuture<JsonNode>> futures =
        tasks.stream()
             .map(c -> CompletableFuture.supplyAsync(() -> {
                     try { return c.call(); }              // wrap checked → runtime
                     catch (Exception e) { throw new CompletionException(e); }
                 }, pool))
             .collect(Collectors.toList());

        List<JsonNode> out = futures.stream()
                                    .map(CompletableFuture::join)   // join each future
                                    .collect(Collectors.toList());

        // Return a single JSON text – either an array or a single object
        return asSegments ? JSON.writeValueAsString(out)
                          : JSON.writeValueAsString(out.get(0));
    }

    // -------- exact‑segment retrieval --------
    private static ObjectNode fetchSegment(String segId, IndexSearcher s) throws Exception {
        Query q = new TermQuery(new Term("id", segId));
        TopDocs td = s.search(q, 1);
        if (td.totalHits.value() == 0) {
            throw new IllegalArgumentException("ID not found: " + segId);
        }
        Document d = s.storedFields().document(td.scoreDocs[0].doc);
        JsonNode raw = JSON.readTree(d.get("raw"));

        ObjectNode out = JSON.createObjectNode();
        out.put("id", segId);
        out.put("segment", raw.path("segment").asText(""));
        return out;
    }

    // -------- full‑document assembly ---------
    private static ObjectNode fetchFull(String segId, DirectoryReader reader) throws Exception {
        String stem = segId.contains("#") ? segId.substring(0, segId.indexOf('#') + 1) : segId + "#"; // ensure trailing #

        List<JsonNode> segs = new ArrayList<>();

        TermsEnum te = reader.leaves().get(0).reader().terms("id").iterator();
        if (te.seekCeil(new BytesRef(stem)) == TermsEnum.SeekStatus.END) {
            throw new IllegalArgumentException("No segments found for " + segId);
        }

        do {
            BytesRef term = te.term();
            if (!StringHelper.startsWith(term, new BytesRef(stem))) break; // prefix exhausted

            PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
            while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                Document d = reader.storedFields().document(pe.docID());
                segs.add(JSON.readTree(d.get("raw")));
            }
        } while (te.next() != null);

        segs.sort(Comparator.comparingInt(n -> n.path("start_char").asInt(0)));

        StringBuilder full = new StringBuilder();
        int currentEnd = -1; // last character already copied (global coordinate)

        for (JsonNode n : segs) {
            String text = n.path("segment").asText("");
            int start = n.path("start_char").asInt(0);
            int end = n.path("end_char").asInt(text.length() + start);

            if (end <= currentEnd) continue; // this segment is wholly inside material we've kept

            int beginInSeg = Math.max(0, currentEnd - start); // skip overlap inside this segment
            if (beginInSeg < text.length()) {
                full.append(text, beginInSeg, text.length()).append(' ');
                currentEnd = end; // advance frontier
            }
        }

        ObjectNode out = JSON.createObjectNode();
        out.put("id", segId);
        out.put("fullText", full.toString());
        return out;
    }
}
