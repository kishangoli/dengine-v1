CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic
NATIVE_EXE = native/dengine-native
NATIVE_SRC = native/main.cpp native/src/model_data.cpp
MODEL_DATA_HEADER = native/include/dengine/model_data.hpp
MODEL_DATA_TEST = native/.build/test_model_data
JSON_HEADER = native/third_party/nlohmann/json.hpp

.PHONY: all test clean

all: $(NATIVE_EXE)

$(NATIVE_EXE): $(NATIVE_SRC) $(MODEL_DATA_HEADER) $(JSON_HEADER)
	$(CXX) $(CXXFLAGS) -I native/include -I native/third_party $(NATIVE_SRC) -o $(NATIVE_EXE)

$(MODEL_DATA_TEST): native/tests/test_model_data.cpp native/src/model_data.cpp $(MODEL_DATA_HEADER)
	mkdir -p native/.build
	$(CXX) $(CXXFLAGS) -I native/include native/tests/test_model_data.cpp native/src/model_data.cpp -o $(MODEL_DATA_TEST)

test: $(NATIVE_EXE) $(MODEL_DATA_TEST)
	python3 native/tests/test_protocol.py $(NATIVE_EXE)
	./$(MODEL_DATA_TEST)

clean:
	rm -f $(NATIVE_EXE)
	rm -rf native/.build
