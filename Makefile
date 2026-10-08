CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic
NATIVE_EXE = native/dengine-native
NATIVE_SRC = native/main.cpp native/src/model_data.cpp native/src/model_loader.cpp
MODEL_DATA_HEADER = native/include/dengine/model_data.hpp
MODEL_LOADER_HEADER = native/include/dengine/model_loader.hpp
MODEL_DATA_TEST = native/.build/test_model_data
MODEL_LOADER_TEST = native/.build/test_model_loader
TOKENIZER_EXE = native/dengine-tokenizer
TOKENIZER_SRC = native/tokenizer_main.cpp native/src/tokenizer.cpp
TOKENIZER_HEADER = native/include/dengine/tokenizer.hpp
JSON_HEADER = native/third_party/nlohmann/json.hpp

.PHONY: all tokenizer test test-model test-tokenizer clean

all: $(NATIVE_EXE)

$(NATIVE_EXE): $(NATIVE_SRC) $(MODEL_DATA_HEADER) $(MODEL_LOADER_HEADER) $(JSON_HEADER)
	$(CXX) $(CXXFLAGS) -I native/include -I native/third_party $(NATIVE_SRC) -o $(NATIVE_EXE)

$(MODEL_DATA_TEST): native/tests/test_model_data.cpp native/src/model_data.cpp $(MODEL_DATA_HEADER)
	mkdir -p native/.build
	$(CXX) $(CXXFLAGS) -I native/include native/tests/test_model_data.cpp native/src/model_data.cpp -o $(MODEL_DATA_TEST)

$(MODEL_LOADER_TEST): native/tests/test_model_loader.cpp native/src/model_data.cpp native/src/model_loader.cpp $(MODEL_DATA_HEADER) $(MODEL_LOADER_HEADER) $(JSON_HEADER)
	mkdir -p native/.build
	$(CXX) $(CXXFLAGS) -I native/include -I native/third_party native/tests/test_model_loader.cpp native/src/model_data.cpp native/src/model_loader.cpp -o $(MODEL_LOADER_TEST)

tokenizer: $(TOKENIZER_EXE)

$(TOKENIZER_EXE): $(TOKENIZER_SRC) $(TOKENIZER_HEADER) $(JSON_HEADER)
	$(CXX) $(CXXFLAGS) -I native/include -I native/third_party $(TOKENIZER_SRC) -licucore -o $(TOKENIZER_EXE)

test: $(NATIVE_EXE) $(MODEL_DATA_TEST) $(MODEL_LOADER_TEST)
	python3 native/tests/test_protocol.py $(NATIVE_EXE)
	./$(MODEL_DATA_TEST)
	./$(MODEL_LOADER_TEST)

test-model: $(NATIVE_EXE)
	python3 native/tests/test_model_inspection.py $(NATIVE_EXE) native/models/smollm2-135m-instruct

test-tokenizer: $(TOKENIZER_EXE)
	native/reference/.venv/bin/python native/tests/test_tokenizer_reference.py $(TOKENIZER_EXE) native/models/smollm2-135m-instruct

clean:
	rm -f $(NATIVE_EXE)
	rm -f $(TOKENIZER_EXE)
	rm -rf native/.build
