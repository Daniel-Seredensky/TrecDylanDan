package src.IR_Ensemble.QA_Assistant.Search;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.io.IOException;
import java.io.InputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class SearcherDaemon {
    private static final Pattern CL_PATTERN =
        Pattern.compile("(?i)^Content-Length:\\s*(\\d+)\\s*$", Pattern.MULTILINE);
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final ExecutorService EXEC = LuceneHolder.getRequestPool();

    /* ---------- helper: framed write ---------- */
    private static void send(JsonNode node, OutputStream rawOut) throws IOException {
        byte[] body = JSON.writeValueAsBytes(node);
        String header = "Content-Length: " + body.length + "\r\n\r\n";
        rawOut.write(header.getBytes(StandardCharsets.UTF_8));
        rawOut.write(body);
        rawOut.flush();               // important when piped
    }

    public static void main(String[] args) throws IOException {

        BufferedInputStream in  = new BufferedInputStream(System.in);
        OutputStream       out = System.out;   // raw, we manage headers ourselves

        /* --- writer queue so responses never interleave --- */
        BlockingQueue<JsonNode> q = new LinkedBlockingQueue<>();
        Thread writer = new Thread(() -> {
            try {
                while (true) send(q.take(), out);
            } catch (InterruptedException | IOException ignored) { }
        }, "daemon-writer");
        writer.setDaemon(true);
        writer.start();

        /* --- read requests with the same framing --- */
        while (true) {
            int contentLen = readContentLength(in);
            if (contentLen < 0) break;     // EOF

            byte[] buf = in.readNBytes(contentLen);
            JsonNode req = JSON.readTree(buf);

            EXEC.submit(() -> {
                ObjectNode resp = JSON.createObjectNode().put("id", req.path("id").asText());
                String call = req.path("call").asText();
                List<String> params = new ArrayList<>();
                req.path("params").forEach(n -> params.add(n.asText()));
                
                try {
                    switch (call) {
                        case "search" -> {
                            Searcher.main(params.toArray(String[]::new));
                            resp.put("status", 0).put("result", "done");
                        }
                        case "selectDocuments" -> {
                            String json = DocumentSelection.run(params);
                            resp.put("status", 0).put("result", json);
                        }
                        default -> throw new IllegalArgumentException("unknown call");
                    }
                } catch (Exception ex) {
                    resp.put("status", 1)
                        .put("exception", ex.getClass().getName())
                        .put("message",   ex.getMessage());
                    ex.printStackTrace(System.err);
                }
                q.add(resp);
            });
        }
        writer.interrupt();
        EXEC.shutdownNow();
    }

    /* ---------- tiny parser for “Content-Length” header ---------- */
    private static int readContentLength(InputStream in) throws IOException {
        ByteArrayOutputStream hdrBuf = new ByteArrayOutputStream(128);
        int b, last = -1;

        /* read until we see either LF LF or CR LF CR LF */
        while ((b = in.read()) != -1) {
            hdrBuf.write(b);

            if (last == '\n' && b == '\n') break;          // matches "\n\n"
            if (hdrBuf.size() >= 4) {
                byte[] a = hdrBuf.toByteArray();
                int n = a.length;
                if (a[n - 4] == '\r' && a[n - 3] == '\n' &&
                    a[n - 2] == '\r' && a[n - 1] == '\n') break; // "\r\n\r\n"
            }
            last = b;
        }

        if (b == -1) return -1;                            // EOF before delimiter

        String headers = hdrBuf.toString(StandardCharsets.UTF_8);
        Matcher m = CL_PATTERN.matcher(headers);
        if (!m.find())
            throw new IOException("Content-Length header not found:\n" + headers);

        return Integer.parseInt(m.group(1));
    }
}
