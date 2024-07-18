if [ "$1" = "debug" ]; then
    BUILD_TYPE="Debug"
    echo "you are running in debug mode ..."
elif [ "$1" = "release" ]; then
    BUILD_TYPE="Release"
    echo "you are running in release mode ..."
else
    echo "Usage: $0 [debug|release]"
    exit 1
fi

mkdir -p build

cd build

# Run CMake with the specified build type and additional optimization for release mode
if [ "$BUILD_TYPE" = "Release" ]; then
    cmake -GNinja -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CXX_FLAGS_RELEASE="-O3" ..
else
    cmake -GNinja -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
fi

ninja

