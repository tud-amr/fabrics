cmake_minimum_required(VERSION 3.12)
project(FabricsCppBridge)

find_package(Threads REQUIRED)

# Compile generated function
add_executable(main src/simulation_loop.cpp)
add_executable(rollout src/rollouts.cpp)
add_executable(multithreading src/multi_threading.cpp)
target_link_libraries(multithreading Threads::Threads)

