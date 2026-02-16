# ============================================================
# Makefile — UCS645 Assignment 3: Correlation
# ============================================================

# Compiler
CXX = g++

# Flags
#   -std=c++17   : C++17 standard
#   -Wall        : enable common warnings
#   -O3          : full optimization (needed for auto-vectorization)
#   -march=native: use all CPU features (AVX2, FMA, etc.) on this machine
#   -fopenmp     : OpenMP multi-threading
#   -mavx2       : explicitly enable AVX2 SIMD
#   -mfma        : enable Fused Multiply-Add
CXXFLAGS = -std=c++17 -Wall -O3 -march=native -fopenmp -mavx2 -mfma

# Target executable
TARGET = correlate

# Source and object files
SOURCES = main.cpp correlate.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = correlate.h

# ============================================================
# Targets
# ============================================================

# Default: build the executable
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each .cpp → .o (depends on headers too)
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================================
# Run with default small matrix (quick sanity check)
# ============================================================
run: $(TARGET)
	./$(TARGET) 200 500

# ============================================================
# Benchmark with increasing matrix sizes
# ============================================================
bench: $(TARGET)
	@echo "=== Small matrix (100 x 500) ==="
	./$(TARGET) 100 500
	@echo ""
	@echo "=== Medium matrix (500 x 1000) ==="
	./$(TARGET) 500 1000
	@echo ""
	@echo "=== Large matrix (1000 x 2000) ==="
	./$(TARGET) 1000 2000

# ============================================================
# perf stat — requires 'perf' installed (sudo apt install linux-tools-generic)
# Usage:  make perf_seq NY=500 NX=1000
#         make perf_par NY=500 NX=1000 THREADS=4
# ============================================================
NY      ?= 500
NX      ?= 1000
THREADS ?= 4

# Wrapper: run the binary but tell main to use only 1 thread (sequential)
perf_seq: $(TARGET)
	perf stat -e task-clock,cycles,instructions,cache-misses,cache-references \
	    ./$(TARGET) $(NY) $(NX) 1

perf_par: $(TARGET)
	perf stat -e task-clock,cycles,instructions,cache-misses,cache-references \
	    ./$(TARGET) $(NY) $(NX) $(THREADS)

# ============================================================
# Thread scaling study
# Run same matrix with 1,2,4,8 threads
# ============================================================
scale: $(TARGET)
	@echo "=== Thread scaling NY=$(NY) NX=$(NX) ==="
	@for t in 1 2 4 8; do \
	    echo "--- $$t thread(s) ---"; \
	    ./$(TARGET) $(NY) $(NX) $$t; \
	done

# ============================================================
# Clean
# ============================================================
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all run bench perf_seq perf_par scale clean