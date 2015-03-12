import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/*
 * @author Ilya Shats
 * @version 1.0
 *
 * C4.5 implementation of a Decision Tree
 */
public class DecisionTree {
    // Node class for the Tree
    static class Node {
        private String value; // class label or attribute value
        private int index; // which column the attribute is in
        private List<Node> children;

        Node(int index, String value) {
            this.index = index;
            this.value = value;
            children = new ArrayList<>();
        }

        public void addChild(Node child) {
            this.children.add(child);
        }
    }

    private static Map<String, String> ops = new HashMap<>(); // map of options, descriptions

    private static boolean header; // false by default
    private static String delim = " ";
    private static int where = 0; // where is the class label, (indexed 0; can use negative values [e.g. -1 is the last]) - by default 0
    private static boolean timeThis; // if true - display execution time on exit - false by default

    private static File trainingSet;
    private static File testingSet;

    private static Node root;
    private static final int ROOT = -2; // used as the index of the root node

    /**
     * Train by building the decision tree
     * @throws IOException
     */
    public static void train() throws IOException {
        List<ArrayList<String>> D = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(trainingSet));
        String line;

        // read in training set
        if (header) br.readLine(); // skip header row
        while ((line = br.readLine()) != null) {
            ArrayList<String> row = new ArrayList<>(Arrays.asList(line.split(delim)));
            // swap the class label with the first column to make life easier later
            // when D decreases in size (subset), will not need to keep track of where
            // the class label is and update "where" accordingly
            if (where != 0) Collections.swap(row, where, 0);
            D.add(new ArrayList<>(Arrays.asList(line.split(delim))));
        }


        root = new Node(ROOT, "");
        generateTree(D, root);
    }

    /**
     * generates the decision tree (recursive)
     * @param D
     * @param parent
     */
    public static void generateTree(List<ArrayList<String>> D, Node parent) {
        // base case 1: no samples left
        if (D.size() == 0) {
            return;
        }

        // base case 2: all classes in D are the same
        String curClass = D.get(0).get(0);
        boolean allSame = true;
        for (ArrayList<String> record : D) {
            if (!record.get(0).equals(curClass)) {
                allSame = false;
                break;
            }
        }

        if (allSame) {
            Node leaf = new Node(-1, curClass);
            parent.addChild(leaf);
            return;
        }

        // base case 3: no attributes left to partition; only class label left
        if (D.get(0).size() == 1) {
            // get class label majority
            String majorityClass = expectedClass(D);
            Node leaf = new Node(-1, majorityClass);
            parent.addChild(leaf);
            return;
        }

        // calculate entropy of the dataset
        double entropy = entropy(D);

        double maxIG = Integer.MIN_VALUE;
        int featIdx = 0;

        // for all attributes in D
        for (int i = 0; i < D.get(0).size(); i++) {
            if (i == 0) continue; // skip the class label

            // compute Information Gain
            double IG = entropy - information(D, i);
            if (IG > maxIG) {
                maxIG = IG;
                featIdx = i;
            }
        }

        // base case 2: no attribute provides any information gain
        //if (maxIG <= 0) {
            //String majorityClass = expectedClass(D);
            //Node leaf = new Node(-1, majorityClass);
            //parent.addChild(leaf);
            //return;
        //}

        parent.index = featIdx;

        final int idx = featIdx;
        Map<String, List<ArrayList<String>>> grouped = D.stream().collect(
                Collectors.groupingBy(list -> list.get(idx)));
        Collection<List<ArrayList<String>>> sublists = grouped.values();

        for (List<ArrayList<String>> sublist : sublists) {
            // create child node
            String attrVal = sublist.get(0).get(featIdx);
            Node child = new Node(-2, attrVal);
            parent.addChild(child);

            // remove attribute we're examining
            for (ArrayList<String> subsublist : sublist) {
                subsublist.remove(featIdx);
            }

            generateTree(sublist, child);
        }
    }

    /**
     * tests the decision tree
     * @param outFile
     * @throws IOException
     */
    public static void test(File outFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(testingSet));
        BufferedWriter out = new BufferedWriter(new FileWriter(outFile));

        int error = 0; // keep track of incorrectly-predicted records
        int N = 0; // number of samples in testingSet
        String line;

        if (header) br.readLine(); // skip header row
        while ((line = br.readLine()) != null) {
            N++;

            List<String> attrs = new ArrayList<>(Arrays.asList(line.split(delim)));
            Node curNode = root;

            // while we haven't reached a leaf node
            while (curNode.index != -1) {
                if (curNode.index < 0) { // if right before leaf
                    curNode = curNode.children.get(0);
                    break;
                }

                String attr = attrs.get(curNode.index);
                attrs.remove(curNode.index);

                // look at children and check the value
                boolean found = false;
                for (Node n : curNode.children) {
                    // find the correct value
                    if (n.value.equals(attr)) {
                        found = true;
                        curNode = n;
                        break;
                    }
                }

                // if the attribute in test didn't match any for some reason(trainingSet didn't cover all cases)
                if (!found) {
                    System.err.println("No node in the decision tree for the attribute value: " + attr);
                    break;
                }
            } // traverse decision tree

            // the value of the current node is the prediction
            String prediction = curNode.value;
            if (!prediction.equals(attrs.get(where))) error++;

            out.write(prediction + "\n");
        }

        double errorRate = (double) error/N;
        double accuracy = (1 - errorRate) * 100;
        out.write(String.format("Accuracy: %.3f%%%n", accuracy));

        br.close();
        out.close();
    }

    /**
     * Returns the mode of the class labels
     * @param D
     * @return
     */
    public static String expectedClass(List<ArrayList<String>> D) {
        Map<String, Integer> counts = new HashMap<>();
        for (ArrayList<String> classLabel : D) {
            String cl = classLabel.get(0);
            if (counts.get(cl) == null) counts.put(cl, 1);
            else counts.computeIfPresent(cl, (String k, Integer v) -> v + 1);
        }

        int majorityClassCount = 0;
        String majorityClass = "";
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > majorityClassCount) {
                majorityClassCount = entry.getValue();
                majorityClass = entry.getKey();
            }
        }

        return majorityClass;
    }


    /**
     * Calculates the entropy of a dataset: -sum{i=1..m} pi*lg(pi); pi is P(Y=yi)
     * @param D
     * @return
     */
    public static double entropy(List<ArrayList<String>> D) {
        Map<String, Double> prior = calcPriorProbs(D);
        double entropy = 0;

        // for each class
        for (double pi : prior.values()) {
            entropy += (pi * log(pi, 2));
        }

        return -entropy;
    }

    /**
     * sum{j=1..v} |Dj|/|D| * entropy(Dj)
     * @param D
     * @param featureIdx
     * @return
     */
    public static double information(List<ArrayList<String>> D, int featureIdx) {
        // figure out how many unique features there are
        Map<String, List<ArrayList<String>>> uniqueFeatures = new HashMap<>();
        Map<String, Integer> counts = new HashMap<>();
        int len = D.size();

        for (ArrayList<String> record : D) {
            String attr = record.get(featureIdx);
            // if this is the first time finding this feature value, add to the map
            if (!uniqueFeatures.containsKey(attr))
                uniqueFeatures.put(attr, new ArrayList<>(len));

            uniqueFeatures.get(attr).add(record);
            if (counts.get(attr) == null) counts.put(attr, 1);
            else counts.computeIfPresent(attr, (String k, Integer v) -> v + 1);
        }

        // sum{j=1,v} (len(Dj)/len(D)*info(Dj))
        double sum = 0;
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            String key = entry.getKey();
            sum += (entropy(uniqueFeatures.get(key)) * counts.get(key) / len);
        }

        return sum;
    }

    /**
     * Calculate priors; used in calculating entropy
     * @param D
     * @return
     */
    public static Map<String, Double> calcPriorProbs(List<ArrayList<String>> D) {
        Map<String, Integer> Y = new HashMap<>(); // Map of classes (counts)
        Map<String, Double> PY = new HashMap<>(); // Map of classes (priors)

        int len = 0; // total number of samples in training set
        // get counts
        for (ArrayList<String> record : D) {
            String yi = record.get(where);
            if (Y.get(yi) == null) Y.put(yi, 1);
            else Y.computeIfPresent(yi, (String k, Integer v) -> v + 1); // update count

            len++;
        }

        // calculate probabilities
        for (Map.Entry<String, Integer> entry : Y.entrySet()) {
            Double p = entry.getValue() / (double) len;
            PY.put(entry.getKey(), p);
        }

        return PY;
    }

    /**
     * log base b of x
     * @param x
     * @param base
     * @return
     */
    public static double log(double x, int base) {
        return Math.log(x)/Math.log(base) + 1e-11;
    }

    /**
     * Sets up the options
     */
    public static void setUpOps() {
        // could use JCommander (http://jcommander.org/)
        ops.put("-h", "header present in data");
        ops.put("-s", "delimiter");
        ops.put("-w", "index of class label in data - most likely 0 or -1 (first or last column)"); // valid flags to NaiveBayes.java
        ops.put("-t", "display execution time");
    }

    /**
     * Parses the options you supply
     * @param args - the list of arguments passed in by the user
     */
    public static void parseOps(String[] args) {
        for (int i = 0; i < args.length - 3; i++) {
            if (args[i].charAt(0) == '-') {
                String op = args[i];
                if (!ops.containsKey(op)) {
                    System.out.printf("%s it not a valid option. Skipped.%n", op);
                    continue;
                }

                switch (op) {
                    case "-h":  // header row in training and testing sets
                        header = true;
                        continue;
                    case "-t": // time
                        timeThis = true;
                        continue;
                    case "-s":  // delimiter
                        i++;
                        delim = args[i];
                        break;
                    case "-w":  // where the class label is (0,1,2...-1,-2,-3, etc.)
                        i++;
                        where = Integer.parseInt(args[i]);
                        break;
                }
            }
        }
    }

    public static void main(String[] args) throws IOException {
        setUpOps();

        if (args.length < 3) {
            System.out.println("USAGE: java DecisionTree [OPTION] TrainingSet TestingSet OutputFile");
            if (ops.size() > 0) {
                System.out.println("OPTIONS:");
                for (Map.Entry<String, String> entry : ops.entrySet()){
                    System.out.println("\t" + entry.getKey() + " - " + entry.getValue());
                }
            }

            System.exit(1);
        }

        parseOps(args);

        int first = args.length-3;
        trainingSet = new File(args[first]);
        testingSet = new File(args[first+1]);
        File out = new File(args[first+2]);

        // start timing here
        long startTime = System.nanoTime();
        train();
        test(out);
        long stopTime = System.nanoTime();

        if (timeThis) {
            long duration = (stopTime - startTime)/1000000; // in ms
            double simpler = 0;
            String unit = "hr";
            if (duration > 1000*60*60) {
                simpler = duration/(1000*60*60.); // hours
            }

            else if (duration > 1000*60) {
                simpler = duration/(1000*60.); // minutes
                unit = "min";
            }

            else if (duration > 1000) {
                simpler = duration/(1000.); // seconds
                unit = "s";
            }

            System.out.printf("Execution time: %d ms%n", duration);
            if (simpler != 0) System.out.printf("\t%.5f %s%n", simpler, unit);
        }
    }
}
