
#include "llvm/CodeGen/ss_section_support.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Value.h"
#include <string>
#include <list>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace llvm {


bool isFunctionCalledBySS (Value* val) {
  std::stringstream tss;
  tss <<  val;
  std::string sVal = tss.str();

  std::ifstream sfile("ss_f.txt");
  if(sfile.fail()){
  	errs() << "ERROR ss_f.txt does not exist \n";
  	return false;
  }


  std::string line;
  for (std::string line; getline( sfile, line);) {
    std::string delimiter = " ";

    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        if (token.compare(sVal) == 0) return true;
        //errs() << token << "\n";
        line.erase(0, pos + delimiter.length());
    }

  }


  return false;
}

}