package src.InfoRetrieval.Search;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.LinkedHashSet;
import java.util.Arrays;

public class SynonymExpander {
    private final Map<String, List<String>> synonymsMap;

    public SynonymExpander(String synonymFilePath) throws IOException {
        this.synonymsMap = loadSynonyms(synonymFilePath);
    }

    private Map<String, List<String>> loadSynonyms(String filePath) throws IOException {
        Map<String, List<String>> map = new HashMap<>();
        List<String> lines = Files.readAllLines(Paths.get(filePath));
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] tokens = line.split(",");
            List<String> entries = new ArrayList<>();
            for (String token : tokens) {
                String term = token.trim();
                if (!term.isEmpty()) entries.add(term);
            }
            for (String term : entries) {
                List<String> syns = new ArrayList<>(entries);
                syns.remove(term);
                map.put(term.toLowerCase(), syns);
            }
        }
        return map;
    }

    public List<String> expandQuery(String query) {
        Set<String> expanded = new LinkedHashSet<>();
        expanded.add(query);
        String[] terms = query.split("\\s+");
        for (int i = 0; i < terms.length; i++) {
            String term = terms[i].toLowerCase();
            List<String> syns = synonymsMap.get(term);
            if (syns != null) {
                for (String syn : syns) {
                    String[] newTerms = Arrays.copyOf(terms, terms.length);
                    newTerms[i] = syn;
                    expanded.add(String.join(" ", newTerms));
                }
            }
        }
        return new ArrayList<>(expanded);
    }
}