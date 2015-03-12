import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
import static java.util.stream.Collectors.collectingAndThen;
import static java.util.stream.Collectors.groupingBy;
import java.util.stream.Collectors;

/**
 * Created by ilya on 2/19/2015.
 */
public class DecisionTree {

    static class Node {
        private String value; // which class
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

    private static List<ArrayList<String>> D = new ArrayList<>();
    private static File trainingSet;
    private static File testingSet;
    private static String delim = " ";
    private static int where = 0;
    private int numAttr = 0;
    private static Node root;

    private static final int ROOT = -2;

    public static void train() throws IOException {
        List<ArrayList<String>> D = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(trainingSet));
        String line;

        // read in training set
        while ((line = br.readLine()) != null)
            D.add(new ArrayList<String>(Arrays.asList(line.split(delim))));

        generateTree(D, root);
    }

    public static String expectedClass(List<ArrayList<String>> D, int where) {
        Map<String, Integer> counts = new HashMap<>();
        for (ArrayList<String> classLabel : D) {
            String cl= classLabel.get(where);
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

    public static void generateTree(List<ArrayList<String>> D, Node parent) {
        // base case 1: no samples left
        if (D.size() == 0) {
            return;
        }

        // base case 2: all classes in D are the same
        String curClass = D.get(0).get(where);
        boolean allSame = true;
        for (ArrayList<String> record : D) {
            if (!record.get(where).equals(curClass)) {
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
            String majorityClass = expectedClass(D, 0);
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
            // will not work in some cases.
            if (i == where) continue; // skip the class label

            // compute Information Gain
            double IG = entropy - information(D, i);
            if (IG > maxIG) {
                maxIG = IG;
                featIdx = i;
            }
        }

        // base case 2: no attribute provides any information gain
        //if (maxIG <= 0) {
            //String majorityClass = expectedClass(D, where);
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
            //System.out.println(parent.index + " " + attrVal);

            // remove attribute we're examining
            for (ArrayList<String> subsublist : sublist) {
                subsublist.remove(featIdx);
            }

            generateTree(sublist, child);
        }
    }

    public static void test(File outFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(testingSet));
        BufferedWriter out = new BufferedWriter(new FileWriter(outFile));
        String line;
        int error = 0;
        int N = 0;

        while ((line = br.readLine()) != null) {
            N++;
            List<String> attrs = new ArrayList<String>(Arrays.asList(line.split(delim)));

            Node curNode = root;
            // while we haven't reached a leaf node
            while (curNode.index != -1) {
                if (curNode.index < 0) {
                    curNode = curNode.children.get(0);
                    break;
                }
                String attr = attrs.get(curNode.index);
                attrs.remove(curNode.index);
                //System.out.println(attr);

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

    public static double entropy(List<ArrayList<String>> D) {
        Map<String, Double> prior = calcPriorProbs(D);
        double entropy = 0;

        // for each class
        for (double pi : prior.values()) {
            entropy += (pi * log(pi, 2));
        }

        return -entropy;
    }

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

    public static Map<String, ArrayList<String>> parseRecord(String record, int classIdx) {
        String[] arr = record.split(delim);
        ArrayList<String> attrs = new ArrayList<String>(Arrays.asList(arr));
        if (classIdx < 0) classIdx += arr.length;
        assert(classIdx >= 0 && classIdx < arr.length);

        Map<String, ArrayList<String>> instance = new HashMap<>();
        instance.put(attrs.remove(classIdx), attrs);

        return instance;
    }

    public static double log(double x, int base) {
        return Math.log(x)/Math.log(base) + 1e-11;
    }

    public static void printTree(Node n) {
        System.out.println(n.value);
        System.out.println(n.index);
        for (Node c : n.children) {
            printTree(c);
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 3) System.exit(1);
        trainingSet = new File(args[0]);
        testingSet = new File(args[1]);
        File out = new File(args[2]);

        delim = "\t";
        where = 0;

        root = new Node(ROOT, "");

        train();
        //printTree(root);
        test(out);
    }
}
