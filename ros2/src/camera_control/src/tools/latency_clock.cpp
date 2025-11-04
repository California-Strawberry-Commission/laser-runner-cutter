#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ASCII font for digits
const std::array<std::array<std::string, 7>, 10> FONT{{
    {" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "},  // 0
    {"  #  ", " ##  ", "# #  ", "  #  ", "  #  ", "  #  ", "#####"},  // 1
    {" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"},  // 2
    {" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "},  // 3
    {"#   #", "#   #", "#   #", "#####", "    #", "    #", "    #"},  // 4
    {"#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "},  // 5
    {" ### ", "#   #", "#    ", "#### ", "#   #", "#   #", " ### "},  // 6
    {"#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "},  // 7
    {" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "},  // 8
    {" ### ", "#   #", "#   #", " ####", "    #", "#   #", " ### "}   // 9
}};

void printDigits(const std::string &digits) {
  std::vector<std::string> lines(7, "");
  for (char c : digits) {
    if (c < '0' || c > '9') {
      continue;
    }

    int idx{c - '0'};
    for (int i = 0; i < 7; ++i) {
      lines[i] += FONT[idx][i] + " ";
    }
  }
  std::cout
      << "\033[2J\033[H\033[1;97m";  // clear screen + move cursor + bright
  for (auto &line : lines) {
    std::cout << line << '\n';
  }
  std::cout << "\033[0m" << std::flush;  // reset format
}

int main() {
  while (true) {
    auto now{std::chrono::system_clock::now()};
    auto timestampMs{std::chrono::duration_cast<std::chrono::milliseconds>(
                         now.time_since_epoch())
                         .count()};

    printDigits(std::to_string(timestampMs));

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}