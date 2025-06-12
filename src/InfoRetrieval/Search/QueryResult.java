package src.InfoRetrieval.Search;

import java.util.List;

public class QueryResult {
    private final String queryId;
    private final List<RankedDocument> documents;

    public QueryResult(String queryId, List<RankedDocument> documents) {
        this.queryId = queryId;
        this.documents = documents;
    }

    public String getQueryId() { return queryId; }
    public List<RankedDocument> getDocuments() { return documents; }
}
