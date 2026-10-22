import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.Random;

public class MatrixOpt {

    // 1. Deterministic Matrix Generator
    public static double[][] generateMatrix(int n, long seed) {
        Random rand = new Random(seed);
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = rand.nextDouble();
            }
        }
        return matrix;
    }

    // 2. Method 1: Baseline Naive (i-j-k)
    public static double[][] multiplyNaive(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // 3. Method 2: Loop Reordering (i-k-j)
    public static double[][] multiplyReordered(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double r = A[i][k];
                for (int j = 0; j < n; j++) {
                    C[i][j] += r * B[k][j];
                }
            }
        }
        return C;
    }

    // 4. Method 3: Cache Blocking / Tiling (Block size: blockSize)
    public static double[][] multiplyBlocked(double[][] A, double[][] B, int n, int blockSize) {
        double[][] C = new double[n][n];
        for (int i0 = 0; i0 < n; i0 += blockSize) {
            int imax = Math.min(i0 + blockSize, n);
            for (int k0 = 0; k0 < n; k0 += blockSize) {
                int kmax = Math.min(k0 + blockSize, n);
                for (int j0 = 0; j0 < n; j0 += blockSize) {
                    int jmax = Math.min(j0 + blockSize, n);

                    // Block computation
                    for (int i = i0; i < imax; i++) {
                        for (int k = k0; k < kmax; k++) {
                            double r = A[i][k];
                            for (int j = j0; j < jmax; j++) {
                                C[i][j] += r * B[k][j];
                            }
                        }
                    }

                }
            }
        }
        return C;
    }

    // 5. Correctness Check Against Baseline
    public static boolean checkCorrectness(int n) {
        double[][] A = generateMatrix(n, 42L);
        double[][] B = generateMatrix(n, 43L);

        double[][] cNaive = multiplyNaive(A, B, n);
        double[][] cReordered = multiplyReordered(A, B, n);
        double[][] cBlocked = multiplyBlocked(A, B, n, 16);

        double epsilon = 1e-9;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(cNaive[i][j] - cReordered[i][j]) > epsilon ||
                    Math.abs(cNaive[i][j] - cBlocked[i][j]) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

    // 6. Execution Benchmark Harness
    public static void main(String[] args) {
        if (!checkCorrectness(30)) {
            System.err.println("Error: Optimization correctness validation failed.");
            return;
        }
        System.out.println("Validation successful: Optimized kernels match naive baseline.");

        int[] sizes = {100, 200, 400, 600, 800, 1000};
        String[] methods = {"Naive_IJK", "Reordered_IKJ", "Blocked_Tiled"};
        int blockSize = 32; // Standard tile fitting in L1/L2 data cache
        int repetitions = 5;
        String csvFile = "dense_optimization_results.csv";

        try {
            boolean writeHeader = !(new File(csvFile).exists());
            PrintWriter writer = new PrintWriter(new FileWriter(csvFile, true));
            if (writeHeader) {
                writer.println("method,n,repetition,time_seconds,memory_mb");
            }

            Runtime runtime = Runtime.getRuntime();

            for (int n : sizes) {
                System.out.printf("\nRunning size n = %d\n", n);

                // Discarded warm-up for JIT stabilization
                double[][] Aw = generateMatrix(n, 1L);
                double[][] Bw = generateMatrix(n, 2L);
                multiplyNaive(Aw, Bw, n);
                multiplyReordered(Aw, Bw, n);
                multiplyBlocked(Aw, Bw, n, blockSize);

                for (String method : methods) {
                    for (int rep = 1; rep <= repetitions; rep++) {
                        double[][] A = generateMatrix(n, 100L + rep);
                        double[][] B = generateMatrix(n, 200L + rep);

                        System.gc();
                        long memBefore = runtime.totalMemory() - runtime.freeMemory();

                        long start = System.nanoTime();
                        double[][] C;
                        if (method.equals("Naive_IJK")) {
                            C = multiplyNaive(A, B, n);
                        } else if (method.equals("Reordered_IKJ")) {
                            C = multiplyReordered(A, B, n);
                        } else {
                            C = multiplyBlocked(A, B, n, blockSize);
                        }
                        long duration = System.nanoTime() - start;

                        long memAfter = runtime.totalMemory() - runtime.freeMemory();
                        double memMb = Math.max(0, memAfter - memBefore) / (1024.0 * 1024.0);
                        if (memMb <= 0.01) {
                            memMb = (double) (n * n * 8L) / (1024.0 * 1024.0);
                        }

                        double elapsedSec = duration / 1.0e9;
                        writer.printf(Locale.US, "%s,%d,%d,%.6f,%.4f\n", method, n, rep, elapsedSec, memMb);
                        writer.flush();
                        System.out.printf("Method: %-15s | n: %4d | Rep: %d | Time: %.4f s\n", method, n, rep, elapsedSec);
                    }
                }
            }

            writer.close();
            System.out.println("\n[Java] Dense optimization benchmark completed successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}