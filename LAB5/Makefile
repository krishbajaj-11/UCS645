CXX = mpicxx
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGETS = q1_daxpy q2_broadcast_race q3_dot_product q4_primes q5_perfect

all: $(TARGETS)

q1_daxpy: q1_daxpy.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q2_broadcast_race: q2_broadcast_race.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q3_dot_product: q3_dot_product.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q4_primes: q4_primes.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q5_perfect: q5_perfect.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)