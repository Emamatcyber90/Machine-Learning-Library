import java.io.*;
import java.util.*;

/**
 * @author Ilya Shats
 * @version 1.0
 *
 * Implementation of Naive-Bayes
 * Works only for categorical variables so far
 */
public class NaiveBayes {
    private static Map<String, String> ops = new HashMap<>(); // map of options, descriptions

    private static boolean debug; // false by default
    private static boolean header; // false by default
    private static String delim = " ";
    private static int where; // where is the class label, (indexed 0; can use negative values [e.g. -1 is the last]) - by default 0
    private static boolean timeThis; // if true - display execution time on exit - false by default
    private static int laplace = 1; // for smoothing

    private static File trainingSet;
    private static File testingSet;

    private static int numAttr; // number of features

    private static Map<String, Integer> Y = new HashMap<>(); // Map of classes (counts)
    private static Map<String, Double> PY = new HashMap<>(); // Map of classes (priors)

    private static Map<String, ArrayList<HashMap<String, Integer>>> posteriorTable = new HashMap<>();

    // currently, confusion matrix is not being used
    private static Map<String, HashMap<String, Integer>> confusionMatrix = new HashMap<>();

    /**
     * Trains the NaiveBayes classifier on trainSet
     * @throws java.io.IOException
     */
    public static void train() throws IOException {
        calcPriorProbs();
        scanTestSet();
        calcPosteriorProbs();
    }

    /**
     * Test the NaiveBayes classifier on testSet and write predictions and classifier
     *  accuracy (100*right/total %) to outFile
     * Add precision and recall? F1 measure?
     * @param outFile the file to which the result is being written
     * @throws IOException
     */
    public static void test(File outFile) throws IOException {
        BufferedReader testing = new BufferedReader(new FileReader(testingSet));
        BufferedWriter out = new BufferedWriter(new FileWriter(outFile));

        int error = 0; // count incorrectly-predicted records
        int N = 0; // number of test samples
        String line;

        // predict for each record
        if (header) testing.readLine(); // skip header row
        while ((line = testing.readLine()) != null) {
            N++;

            Map.Entry<String, ArrayList<String>> record = parseRecord(line, where).entrySet().iterator().next(); // (line, class index)
            String yi = record.getKey();
            ArrayList<String> x = record.getValue();

            // maxarg((prod{k=1..n} P(Xk | Yi)) * P(Yi) for all Yi in Y)
            double max = 0;
            String maxarg = "";

            // (prod{k=1..n} P(Xk | Yi)) * P(Yi) for all Yi in Y
            // should actually do ln(P(Yi)) + sum{k=1..n} ln(P(Xk | Yi)) for all Yi in Y to avoid floating point underflow
            for (Map.Entry<String, Double> entry : PY.entrySet()) {
                String c = entry.getKey(); // current class
                int total = Y.get(c);
                double product = entry.getValue(); // init to prior - P(Yi); here Yi == c

                // for each attribute, calculate P(Xk | Yi) and compute their product
                for (int k = 0; k < x.size(); k++) {
                    String attr = x.get(k);
                    // count number of times attr_k appears in training set when appropriate class is present
                    int count = posteriorTable.get(c).get(k).get(attr)+laplace; // Laplace smoothing
                    product *= (double) count/total;

                    if (debug) System.out.printf("P(%s | %s) = %f/%d = %.3f%n", attr, c, count, total, count/total);
                }

                if (debug) System.out.printf("Product for class %s: %.3f%n", c, product);

                if (product > max) {
                    max = product;
                    maxarg = c;
                }
            }

            // error
            if (!maxarg.equals(yi)) error++;

            // to remind myself, columns are predicted values, rows are actual values
            // in this case, first val I pass (outer Map) is the actual value
            confusionMatrix.get(yi).computeIfPresent(maxarg, (String k, Integer i) -> i + 1);

            // write predicted class to file
            out.write(maxarg + "\n");
        }

        double errorRate = (double) error/N;
        double accuracy = (1 - errorRate) * 100; // accuracy
        out.write(String.format("Accuracy: %.3f%%%n", accuracy));

        // precision, recall (sensitivity), specificity, F1 score, ...

        // close Readers and Writers
        testing.close();
        out.close();
    }

    /**
     * Calculate the prior probabilities (P(Yi)) from trainSet
     * @throws java.io.IOException
     */
    public static void calcPriorProbs() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(trainingSet));
        String line;
        int len = 0; // total number of samples in training set

        // get counts
        if (header) br.readLine(); // skip header row
        while ((line = br.readLine()) != null) {
            // get class of current record (Yi)
            Map.Entry<String, ArrayList<String>> record = parseRecord(line, where).entrySet().iterator().next(); // (line, class index)
            String yi = record.getKey();

            // put all classes in the table of posterior probabilities and confusionMatrix
            if (!posteriorTable.containsKey(yi)) {
                posteriorTable.put(yi, new ArrayList<>(numAttr));
                confusionMatrix.put(yi, new HashMap<>()); // set up confusionMatrix for later
                confusionMatrix.get(yi).put(yi, 0);
            }

            // record class and count
            if (Y.get(yi) == null) Y.put(yi, 1);
            else Y.computeIfPresent(yi, (String k, Integer v) -> v + 1); // update count

            // keep track of total length
            len++;
        }

        // calculate probabilities
        for (Map.Entry<String, Integer> entry : Y.entrySet()) {
            Double p = entry.getValue() / (double) len;
            PY.put(entry.getKey(), p);

            if (debug) System.out.printf("P(%s) = %.3f%n", entry.getKey(), p);
        }

        br.close();
    }

    /**
     * After posteriorTable is set up, calculate posteriors and stick in the table.
     *  Note: technically these aren't posterior probabilities, but the
     *  last step is simple - divide by number of class_k instances
     *  I do it later to prevent error propagation: instead of 2/3*5/7, do (2*5)/(3*7)
     * @throws IOException
     */
    public static void calcPosteriorProbs() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(trainingSet));
        String line;

        if (header) br.readLine(); // skip header row
        while ((line = br.readLine()) != null) {
            Map.Entry<String, ArrayList<String>> record = parseRecord(line, where).entrySet().iterator().next(); // (line, class index)
            String yi = record.getKey();
            ArrayList<String> attrs = record.getValue();

            // iterate through attributes
            for (int i = 0; i < numAttr; i++) {
                Map<String, Integer> hm = posteriorTable.get(yi).get(i);
                hm.computeIfPresent(attrs.get(i), (String k, Integer v) -> v + 1); // update count
            }
        }
    }

    /**
     * Scans the testing set once and set up the posteriorTable
     * The idea is, if you're never testing a record with feature xi, don't stick
     *  it in the table. No need to calculate its posterior probability and waste time
     * @throws IOException
     */
    public static void scanTestSet() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(testingSet));
        String line;

        if (header) br.readLine(); // skip header row
        while ((line = br.readLine()) != null) {
            Map.Entry<String, ArrayList<String>> record = parseRecord(line, where).entrySet().iterator().next(); // (line, class index)
            ArrayList<String> attrs = record.getValue();

            // need to add the attribute to EACH class (for calc'ing posteriors)
            for (Map.Entry<String, ArrayList<HashMap<String, Integer>>> entry : posteriorTable.entrySet()) {
                if (!posteriorTable.containsKey(entry.getKey())) {
                    System.err.printf("\"%s\" is not a valid class. " +
                                    "Class found in testing set that is not in training set. Skipped.%n",
                            entry.getKey());
                    continue;
                }


                ArrayList<HashMap<String, Integer>> list = entry.getValue();
                // for each attribute, add unique values to table
                for (int i = 0; i < numAttr; i++) {
                    // if it doesn't have a HM yet, add one
                    if (list.size() <= i) list.add(new HashMap<>());
                    list.get(i).put(attrs.get(i), 0);
                }
            } // for each class
        }
    }

    /**
     * Returns an instance (Map) of the record. k,v -> class label, attributes
     * @param record - the record containing the class label and the features
     * @param classIdx - where the class label is (indexed at 0)
     * @return Map - key is class label, value is list of features
     */
    public static Map<String, ArrayList<String>> parseRecord(String record, int classIdx) {
        String[] arr = record.split(delim);
        ArrayList<String> attrs = new ArrayList<>(Arrays.asList(arr));
        if (classIdx < 0) classIdx += arr.length;
        assert(classIdx >= 0 && classIdx < arr.length);

        Map<String, ArrayList<String>> instance = new HashMap<>();
        instance.put(attrs.remove(classIdx), attrs);

        return instance;
    }

    /**
     * Returns the number of attributes/features you're training on
     * @return int - the number of attributes
     * @throws IOException
     */
    public static int getNumAttr() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(trainingSet));
        if (header) br.readLine();

        String line = br.readLine();
        if (line == null) die("Nothing in test set");

        String[] tokens = line.split(delim);
        br.close();

        return tokens.length - 1;
    }

    /**
     * Sets up the options
     */
    public static void setUpOps() {
        // could use JCommander (http://jcommander.org/)
        ops.put("-d", "debug info (totally useless; see comment)");
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

                /*
                The debug option is (virtually) useless.
                It prints some values at intermediate steps (priors, posteriors, pre-posteriors...)
                If you don't trust the algorithm, you are (obviously) free to test it on datasets to check
                if the probabilities are correct (which is why I kept the option).
                However, use a small dataset, otherwise it will spam your stdout, and you will be super annoyed.
                Also, the output is very ugly; linear and almost nonsensical; requires human parsing :(
                 */
                switch (op) {
                    case "-d":  // debug
                        debug = true;
                        continue;
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

    /**
     * If a fatal error occurs, call this to kill the program
     * @param msg - the message to display before dying
     */
    public static void die(String msg) {
        System.err.println(msg);
        System.exit(1);
    }

    public static void main(String[] args) throws IOException {
        setUpOps();

        if (args.length <= 2) {
            System.out.println("USAGE: java NaiveBayes [OPTION] TrainingSet TestingSet OutputFile");
            if (ops.size() > 0) {
                System.out.println("OPTIONS:");
                for (Map.Entry<String, String> entry : ops.entrySet()){
                    System.out.println("\t" + entry.getKey() + " - " + entry.getValue());
                }
            }

            System.exit(1);
        }

        parseOps(args);

        // open files; validates them before starting training and then crashing when testingSet doesn't exist
        int first = args.length-3;
        trainingSet = new File(args[first]);
        testingSet = new File(args[first+1]);
        File outFile = new File(args[first+2]);

        // start timing here
        long startTime = System.nanoTime();
        numAttr = getNumAttr(); // get the number of attributes

        // train and test
        train();
        test(outFile);
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

            /*
            Had to make a decision here.
            Representing duration as 5.3 minutes instead of 5 minutes and x seconds is useful for plotting (just plot 5.3 rather
            than converting x seconds). On the other hand, not as nice for readability.

            Now, I could provide the option to display in one format or the other, but people don't want to pass in 50 different
            options. I'll just display the floating point format.
             */

            System.out.printf("Execution time: %d ms%n", duration);
            if (simpler != 0) System.out.printf("\t%.5f %s%n", simpler, unit);
        }
    }
}
