#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Constants.h"
#include<string>
#include <list>
#include <fstream>
#include <iostream>

using namespace llvm;

extern std::list<Value *> funcsCalledBySSDirectly;
extern std::list<std::string> AllFuncsCalledBySS;
extern std::list<Value *> tmpFuncList;

extern std::ofstream outfile;

// extern std::list<Value *> ptrFuncsCalledBySSDirectly;
// extern std::list<Value *> ptrTmpFuncList;

//bool isFunctionCalledBySS (Value* val);