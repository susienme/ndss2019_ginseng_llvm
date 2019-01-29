#include "llvm/Support/ss_section_support.h"
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
#include "llvm/IR/SSPassData.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace llvm;

bool isFunctionCalledBySS (Value* val);