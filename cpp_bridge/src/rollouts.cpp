#include <iostream>
#include "planner.c"
#include <chrono>

unsigned int DOF = 7;

void print_array(const char* intro, double* array, unsigned int length) {
  std::cout << intro;
  for(unsigned int i = 0; i < length; ++i){
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

void print_array(const char* intro, const double* array, unsigned int length) {
  std::cout << intro;
  for(unsigned int i = 0; i < length; ++i){
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

void integrate(double** input, double* action, double time_step) {
  for (unsigned int i = 0; i < DOF; i++) {
    input[0][i] = input[1][i] * time_step;
    input[1][i] = action[i] * time_step;
  }
}
  

void compute_rollout(
    double** input,
    unsigned int horizon,
    double time_step,
    double** output
) {
  long long int* setting_0 = new long long int[2];
  double* setting_1 = new double[2];
  int setting_2 = 0;
  for (unsigned int i = 0; i < horizon; i++) {
    const double** const_input = const_cast<const double**>(input);
    integrate(input, output[0], time_step);
    casadi_f0(const_input, output, setting_0, setting_1, setting_2);
    //print_array("Joint Positions : ", input[0], DOF);
    //print_array("Joint Velocities : ", input[1], DOF);
    double test = output[0][0] * time_step;
    
  }
}

int main(int argc, char *argv[])
{
  // Some settings
  // Unclear what they are used for

  // Define inputs and outputs
  // Potential Segmentation Faults can be caused by the wrong number of arguments.
  unsigned int nb_inputs = 6;
  unsigned int nb_outputs = 1;

  double joint_positions[] = {0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1};
  double joint_velocities[] = {0.4, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1};
  double goal_position_0[] = {0.3, 0.2, 0.1};
  double weight_goal_0[] = {0.1};
  double goal_position_1[] = {0.0, 0.0, 0.107};
  double weight_goal_1[] = {0.1};

  double** input = new double*[nb_inputs];
  input[0] = joint_positions;
  input[1] = joint_velocities;
  input[2] = weight_goal_0;
  input[3] = goal_position_0;
  input[4] = weight_goal_1;
  input[5] = goal_position_1;
  double** output = new double*[nb_outputs];
  output[0] = new double[DOF];

  // Timing 
  auto start = std::chrono::high_resolution_clock::now();

  double time_step = atof(argv[2]);
  unsigned int horizon = atoi(argv[1]);
  compute_rollout(input, horizon, time_step, output);

  // Stop the timer
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time in microseconds
  std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>((end - start));

  // Print the elapsed time in microseconds, milliseconds, or seconds
  std::cout << "Rollout computed in: " << duration.count() * 0.001 << " milliseconds" << std::endl;
  return EXIT_SUCCESS;
}
