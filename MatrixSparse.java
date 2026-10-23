import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

public class MatrixSparse {

    // 1. CSR Representation
    public static class CSRMatrix {
        public int rows;
        public int cols;
        public double[] values;
        public int[] colIndex;
        public int[] rowPointer;

        public CSRMatrix(int rows, int cols, double[] values, int[] colIndex, int[] rowPointer) {
            this.rows = rows;
            this.cols = cols;
            this.values = values;
            this.colIndex = colIndex;
            this.rowPointer = rowPointer;
        }

        public int getNonZeros() {
            return values.length;
        }
    }

    // 2. Synthetic Sparse Matrix Generator (returns both dense and CSR)
    public static double[][] generateDenseSparse(int n, double sparsity, long seed) {
        Random rand = new Random(seed);
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (rand.nextDouble() >= sparsity) {
                    matrix[i][j] = rand.nextDouble() + 0.1; // Non-zero entry
                } else {
                    matrix[i][j] = 0.0;
                }
            }
        }
        return matrix;
    }

    public static CSRMatrix denseToCSR(double[][] dense, int n) {
        ArrayList<Double> valList = new ArrayList<>();
        ArrayList<Integer> colList = new ArrayList<>();
        int[] rowPointer = new int[n + 1];

        int count = 0;
        rowPointer[0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dense[i][j] != 0.0) {
                    valList.add(dense[i][j]);
                    colList.add(j);
                    count++;
                }
            }
            rowPointer[i + 1] = count;
        }

        double[] values = new double[valList.size()];
        int[] colIndex = new int[colList.size()];
        for (int i = 0; i < valList.size(); i++) {
            values[i] = valList.get(i);
            colIndex[i] = colList.get(i);
        }

        return new CSRMatrix(n, n, values, colIndex, rowPointer);
    }

    // 3. Dense Multiplication Baseline (Optimized Reordered IKJ)
    public static double[][] multiplyDense(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double r = A[i][k];
                if (r != 0.0) {
                    for (int j = 0; j < n; j++) {
                        C[i][j] += r * B[k][j];
                    }
                }
            }
        }
        return C;
    }

    // 4. Sparse CSR x CSR Multiplication Kernel
    public static CSRMatrix multiplySparse(CSRMatrix A, CSRMatrix B, int n) {
        ArrayList<Double> resValues = new ArrayList<>();
        ArrayList<Integer> resCols = new ArrayList<>();
        int[] resRowPointer = new int[n + 1];

        double[] accumulator = new double[n];
        int[] marker = new int[n];
        for (int idx = 0; idx < n; idx++) {
            marker[idx] = -1;
        }

        resRowPointer[0] = 0;
        int nonZeroCounter = 0;

        for (int i = 0; i < n; i++) {
            int head = -2;
            int aStart = A.rowPointer[i];
            int aEnd = A.rowPointer[i + 1];

            for (int aIdx = aStart; aIdx < aEnd; aIdx++) {
                double aVal = A.values[aIdx];
                int k = A.colIndex[aIdx];

                int bStart = B.rowPointer[k];
                int bEnd = B.rowPointer[k + 1];

                for (int bIdx = bStart; bIdx < bEnd; bIdx++) {
                    int j = B.colIndex[bIdx];
                    double bVal = B.values[bIdx];

                    accumulator[j] += aVal * bVal;
                    if (marker[j] == -1) {
                        marker[j] = head;
                        head = j;
                    }
                }
            }

            // Extract accumulated non-zeros for row i
            while (head != -2) {
                int j = head;
                head = marker[j];
                marker[j] = -1;

                double sum = accumulator[j];
                accumulator[j] = 0.0;

                if (Math.abs(sum) > 1e-12) {
                    resValues.add(sum);
                    resCols.add(j);
                    nonZeroCounter++;
                }
            }
            resRowPointer[i + 1] = nonZeroCounter;
        }

        double[] outValues = new double[resValues.size()];
        int[] outCols = new int[resCols.size()];
        for (int idx = 0; idx < resValues.size(); idx++) {
            outValues[idx] = resValues.get(idx);
            outCols[idx] = resCols.get(idx);
        }

        return new CSRMatrix(n, n, outValues, outCols, resRowPointer);
    }

    // 5. Correctness Check (Sparse vs Dense)
    public static boolean checkCorrectness(int n, double sparsity) {
        double[][] denseA = generateDenseSparse(n, sparsity, 42L);
        double[][] denseB = generateDenseSparse(n, sparsity, 43L);

        CSRMatrix sparseA = denseToCSR(denseA, n);
        CSRMatrix sparseB = denseToCSR(denseB, n);

        double[][] denseC = multiplyDense(denseA, denseB, n);
        CSRMatrix sparseC = multiplySparse(sparseA, sparseB, n);

        // Convert sparseC back to dense for comparison
        double[][] check = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int idx = sparseC.rowPointer[i]; idx < sparseC.rowPointer[i + 1]; idx++) {
                check[i][sparseC.colIndex[idx]] = sparseC.values[idx];
            }
        }

        double epsilon = 1e-9;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(denseC[i][j] - check[i][j]) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

    // 6. Benchmark Harness
    public static void main(String[] args) {
        if (!checkCorrectness(50, 0.90)) {
            System.err.println("Error: Sparse multiplication validation failed.");
            return;
        }
        System.out.println("Validation successful: CSR multiplication matches dense baseline.");

        int n = 500; // Fixed matrix dimension
        double[] sparsities = {0.50, 0.80, 0.90, 0.95, 0.98, 0.99};
        int repetitions = 5;
        String csvFile = "sparse_results.csv";

        try {
            boolean writeHeader = !(new File(csvFile).exists());
            PrintWriter writer = new PrintWriter(new FileWriter(csvFile, true));
            if (writeHeader) {
                writer.println("method,n,sparsity,density,repetition,time_seconds,memory_mb,nnz_a,nnz_b,nnz_out");
            }

            for (double sp : sparsities) {
                double density = 1.0 - sp;
                System.out.printf("\nTesting Sparsity: %.0f%% (Density: %.0f%%) for n = %d\n",
                                  sp * 100, density * 100, n);

                // Warm-up run
                double[][] Aw = generateDenseSparse(n, sp, 1L);
                double[][] Bw = generateDenseSparse(n, sp, 2L);
                CSRMatrix swA = denseToCSR(Aw, n);
                CSRMatrix swB = denseToCSR(Bw, n);
                multiplyDense(Aw, Bw, n);
                multiplySparse(swA, swB, n);

                for (int rep = 1; rep <= repetitions; rep++) {
                    double[][] A = generateDenseSparse(n, sp, 100L + rep);
                    double[][] B = generateDenseSparse(n, sp, 200L + rep);

                    CSRMatrix spA = denseToCSR(A, n);
                    CSRMatrix spB = denseToCSR(B, n);

                    // 1. Measure Dense
                    long startDense = System.nanoTime();
                    double[][] C_dense = multiplyDense(A, B, n);
                    long durationDense = System.nanoTime() - startDense;
                    double timeDenseSec = durationDense / 1.0e9;

                    double memDenseMb = (3.0 * n * n * 8.0) / (1024.0 * 1024.0);
                    writer.printf(Locale.US, "Dense_IKJ,%d,%.2f,%.2f,%d,%.6f,%.4f,%d,%d,%d\n",
                            n, sp, density, rep, timeDenseSec, memDenseMb, spA.getNonZeros(), spB.getNonZeros(), n * n);

                    // 2. Measure Sparse CSR
                    long startSparse = System.nanoTime();
                    CSRMatrix C_sparse = multiplySparse(spA, spB, n);
                    long durationSparse = System.nanoTime() - startSparse;
                    double timeSparseSec = durationSparse / 1.0e9;

                    // CSR Memory: (values: 8B * nnz) + (colIndex: 4B * nnz) + (rowPointer: 4B * (n + 1))
                    long totalBytesSparse = (long) (spA.getNonZeros() + spB.getNonZeros() + C_sparse.getNonZeros()) * 12L
                                            + 3L * (n + 1) * 4L;
                    double memSparseMb = totalBytesSparse / (1024.0 * 1024.0);

                    writer.printf(Locale.US, "Sparse_CSR,%d,%.2f,%.2f,%d,%.6f,%.4f,%d,%d,%d\n",
                            n, sp, density, rep, timeSparseSec, memSparseMb, spA.getNonZeros(), spB.getNonZeros(), C_sparse.getNonZeros());

                    writer.flush();
                    System.out.printf("Rep %d | Dense: %.4f s (%.2f MB) | Sparse: %.4f s (%.2f MB) | Output NNZ: %d\n",
                            rep, timeDenseSec, memDenseMb, timeSparseSec, memSparseMb, C_sparse.getNonZeros());
                }
            }

            writer.close();
            System.out.println("\n[Java] Sparse matrix benchmark completed successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}