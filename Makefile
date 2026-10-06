CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic
NATIVE_EXE = native/dengine-native
NATIVE_SRC = native/main.cpp
JSON_HEADER = native/third_party/nlohmann/json.hpp

.PHONY: all test clean

all: $(NATIVE_EXE)

$(NATIVE_EXE): $(NATIVE_SRC) $(JSON_HEADER)
	$(CXX) $(CXXFLAGS) -I native/third_party $(NATIVE_SRC) -o $(NATIVE_EXE)

test: $(NATIVE_EXE)
	python3 native/tests/test_protocol.py $(NATIVE_EXE)

clean:
	rm -f $(NATIVE_EXE)
