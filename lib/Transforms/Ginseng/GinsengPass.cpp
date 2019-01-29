
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include<string>
#include "llvm/CodeGen/ginseng.h"
#include "llvm/IR/IRBuilder.h"

#define SS_DATA_ANNOTATION_STR   "SS_DATA\00"
#define UUIDGEN_CMD     "call"

using namespace llvm;

const int NO_OF_SS_DATA = 7;

std::string ss_data_annotation = "SS_DATA";

static StringRef getGlobalStringConstant(raw_ostream& O, Value *val) {
  if (val->getValueID() != Value::ConstantExprVal) {
    return StringRef();
  }
  ConstantExpr *ce = (ConstantExpr *)val;
  GlobalVariable *gv = (GlobalVariable *)ce->getOperand(0);

  return cast<ConstantDataArray>(gv->getInitializer())->getAsString();
}


namespace {
  void printMAP(std::map<Value *, Value *> *pMap) {
    for(auto &pair : *pMap) {
      pair.first->print(ymh_log() << "ALLOCA: " );
      errs() << " : ";
      pair.second->print(errs());
      errs() << "\n";
    }
  }

  void printChars(StringRef &str) {
    ymh_log() << "printing: " << str << " with size of " << str.size() << "\n";
    for(size_t i = 0; i < str.size(); i++) {
      ymh_log() << i << ": " << str.data()[i] << " " << ((int) str.data()[i]) <<"\n";
    }
  }

  // copied from "AsmWriter.cpp"
  bool isReferencingMDNode(const Instruction &I) {
    if (const auto *CI = dyn_cast<CallInst>(&I))
      if (Function *F = CI->getCalledFunction())
        if (F->isIntrinsic())
          for (auto &Op : I.operands())
            if (auto *V = dyn_cast_or_null<MetadataAsValue>(Op))
              if (isa<MDNode>(V->getMetadata()))
                return true;
    return false;
  }

  struct SkeletonPass : public FunctionPass {
    static char ID;
    Value* m_ssLocalVarsArgs [NO_OF_SS_DATA];
    Value* m_ssArgs [NO_OF_SS_DATA];

    std::vector<std::string> m_debugFuncs;
    std::vector<std::string> m_slibFuncs;

    SkeletonPass() : FunctionPass(ID) {
      m_debugFuncs.push_back("printf");
      m_debugFuncs.push_back("printRegs");

      m_slibFuncs.push_back(SLIB_FUNC_SS_READ);
      m_slibFuncs.push_back(SLIB_FUNC_SS_WRITE);
      m_slibFuncs.push_back(SLIB_FUNC_SS_SAVE_CLEAN_V);
      m_slibFuncs.push_back(SLIB_FUNC_SS_READ_V);
      m_slibFuncs.push_back(SLIB_FUNC_SS_SAVE_M);
      m_slibFuncs.push_back(SLIB_FUNC_SS_READ_M);
    }

    bool isDebugFunc(Function *pFn) {
      std::string fnName = pFn->getName().str();
      for (std::string _fnName: m_debugFuncs) {
        if (!fnName.compare(_fnName)) return true;
      }

      return false;
    }

    bool isLLVMFunc(Function *pFn) {
      return false; 
      std::string fnName = pFn->getName().str();
      const char llvmPrefix[] = "llvm.";

      if (fnName.size() < strlen(llvmPrefix)) return false;

      if (!fnName.compare(0, strlen(llvmPrefix), llvmPrefix)) return true;

      return false; 
    }

    bool isSlibFunc(Function *pFn) {
      std::string fnName = pFn->getName().str();
      for (std::string _fnName: m_slibFuncs) {
        if (!fnName.compare(_fnName)) return true;
      }

      return false; 
    }

    void clearSSList() {
      bzero(m_ssLocalVarsArgs, NO_OF_SS_DATA * sizeof(void *));
      bzero(m_ssArgs, NO_OF_SS_DATA * sizeof(void *));
    }

    void checkMD(Function &F) {
      Module *M = F.getParent();
      GlobalVariable *pGV = M->getNamedGlobal("llvm.global.annotations");
      ymh_log() << "CHECK_MD: Working on " << F.getName() << "\n";
      if (pGV) {
        ConstantArray *arr = cast<ConstantArray>(pGV->getOperand(0));
        for (unsigned int i = 0; i < arr->getNumOperands(); i++) {
          ConstantStruct *constStruct = cast<ConstantStruct>(arr->getOperand(i));

          if (Function *fn = dyn_cast<Function>(constStruct->getOperand(0)->getOperand(0))) {
            if (fn == &F) ymh_log() << "CHECK_MD: This is an annotated function" << "\n";
          }
        }
      }
    }

    bool inArgs(Value *v) {
      for (int i = 0; i < NO_OF_SS_DATA; i++) {
        if (m_ssArgs[i] == v) return true;
      }
      return false;
    }

    bool printSSVarArg(Function &F) {
      errs() << "\n";

      if (!m_ssLocalVarsArgs[0]) {
        ymh_log() << F.getName() <<"() has no SS data\n";
        return false;
      }

      ymh_log() << F.getName() <<"():\n";
      for (int i = 0; i < NO_OF_SS_DATA; i++) {
        if (m_ssLocalVarsArgs[i]) {
          if (!inArgs(m_ssLocalVarsArgs[i])) {
            m_ssLocalVarsArgs[i]->print(ymh_log() << YMH_COLOR_GREEN << "SS_LOCAL_VAR"); errs() << YMH_COLOR_RESET << "\n";
          } else {
            m_ssLocalVarsArgs[i]->print(ymh_log() << YMH_COLOR_YELLOW << "SS_LOCAL_ARG"); errs() << YMH_COLOR_RESET << "\n";
          }
        }
      }

      return true;
    }

    void printLL(Function &F) {
      ymh_log() << F.getName() <<"():\n";
      for (auto &BB: F) {
        for (auto &I: BB) {
          I.print(ymh_log() << YMH_COLOR_BRIGHT_BLACK);
          errs() << YMH_COLOR_RESET << "\n";
        }
      }
    }

    MDNode* getSSAnnotation(Instruction &I) {
      const std::string annotation = "SS_VAR";
      MDNode *pMDNode;
      if ( I.hasMetadata() ) {
        // non-postfixed
        if ( (pMDNode = I.getMetadata(annotation)) ) return pMDNode;

        // postfixed
        for (int i = 0; i < NR_MAX_SS_DATA_VAR; i++) {
          std::string postfixedAnnon = annotation + std::to_string(i);
          if ( (pMDNode = I.getMetadata(postfixedAnnon)) ) {
            return pMDNode;
          }
        }
      }

      return nullptr;
    }
    
    bool defineSReg(Function *F) {
      for(auto &BB : *F) {
        for(auto &I : BB) {
          MDNode *pMDNode = getSSAnnotation(I);
          if (pMDNode == nullptr) continue;

          // if 'alloca' -> replace it with 
        }
      }

      return false;
    }

    bool inSSArgs(Value *v) {
      for(int i = 0; i < NR_MAX_SS_DATA_VAR; i++) {
        if (v == m_ssArgs[i]) return true;
      }

      return false;
    }

    MDNode* getMDNode(Instruction &I) {
      LLVMContext& C = I.getContext();
      return MDNode::get(C, MDString::get(C, "SS_DATA"));
    }

    void addMetadataVAR(Instruction &I, int idx) {
      char SS_VAR[8];
      MDNode* N = getMDNode(I);

      sprintf(SS_VAR, "SS_VAR%d", idx);
      I.setMetadata(SS_VAR, N);
    }

    void addMetadataARG(Instruction &I, int idx) {
      char SS_ARG[8];
      MDNode* N = getMDNode(I);

      sprintf(SS_ARG, "SS_ARG%d", idx);
      I.dropUnknownNonDebugMetadata();  // this drops SS_VAR
      I.setMetadata(SS_ARG, N);
    }

    bool inArgInFunc(CallInst *pCallInst, Instruction *pI) {
      // bool bFound = false;
      for (unsigned i = 0; i < pCallInst->getNumArgOperands(); i++) {
        Instruction *pArgInstr = dyn_cast<Instruction>(pCallInst->getArgOperand(i));
        if (pArgInstr && pArgInstr == pI) {
          pI->print(ymh_log() << "XCALL: found");
          errs() << " for " << pCallInst->getCalledFunction()->getName() << "()\n";
          return true;
        }
      }

      return false;
    }

    void insertBeforeLastArgLoad(BasicBlock::iterator &curIt, BasicBlock &B, Function *pCallee, 
                                Instruction *newInstr, const char* name = "") {
      BasicBlock::iterator lastLoad = curIt;
      CallInst *pCallInst = dyn_cast<CallInst>(curIt);

      std::vector<Value *> args;
      std::vector<bool> found;
      const unsigned unfoundArgs = pCallInst->getNumArgOperands();
      unsigned nrArgsWithLoad = 0;

      // list up all args
      if (pCallee) ymh_log() << "ARG_CHECK: Calling " << pCallee->getName() << " needs " << unfoundArgs << " args\n";
      else ymh_log() << "ARG_CHECK: Calling a function pointer needs " << unfoundArgs << " args\n";
      for (unsigned i = 0; i < unfoundArgs; i++) {
        Value *pArg = pCallInst->getOperand(i);
        if (isa<LoadInst>(pArg)) {
          args.push_back(pArg);
          found.push_back(false);
          nrArgsWithLoad++;
        }
      }
      ymh_log() << "ARG_CHECK: " << nrArgsWithLoad << "/" << unfoundArgs << " are load args\n";
      if (!nrArgsWithLoad) {
        lastLoad->print(ymh_log() << "Inserting " << name << " before");
        errs() << "\n";
        B.getInstList().insert(lastLoad, newInstr);
        return;
      }

      // find load
      while(1) {
        lastLoad--;
        Instruction *pInst;

        for (unsigned i = 0; i < args.size(); i++ ){
          pInst = dyn_cast<Instruction>(lastLoad);
          if (args[i] == pInst && !found[i]) {
            found[i] == true;
            nrArgsWithLoad--;

            pInst->print(ymh_log() << "Found:" << YMH_COLOR_BRIGHT_BLACK);
            errs() << YMH_COLOR_RESET << "\n";
          }

          if (!nrArgsWithLoad) {
            ymh_log() << "ARG_CHECK: all args are found\n";
            /*lastLoad->print( ymh_log() << "The last arg: " << YMH_COLOR_BRIGHT_BLACK );
            errs() << YMH_COLOR_RESET << "\n";*/

            lastLoad->print(ymh_log() << "Inserting " << name << " before");
            errs() << "\n";
            B.getInstList().insert(lastLoad, newInstr);
            return;
          }
        }
      
        if (lastLoad == B.begin()) {
          ymh_log() << "Cannot find all load args...\n";
          exit(1);
        }
      }

    }

    virtual bool runOnFunction(Function &F) {
      checkMD(F);
      clearSSList();
      Module *M = F.getParent();
      bool bModified = false;
      
      if (!M->m_pFnSSSaveCleanV) {
        // Make the function type:  double(double,double) etc.
        std::vector<Type*> argTypes_2(2, Type::getInt64Ty(F.getContext()));
        std::vector<Type*> argTypes_3(3, Type::getInt64Ty(F.getContext()));
        std::vector<Type*> argTypes_4(4, Type::getInt64Ty(F.getContext()));

        FunctionType *FT_2 = FunctionType::get(Type::getInt64Ty(F.getContext()), argTypes_2, false);
        FunctionType *FT_3 = FunctionType::get(Type::getInt64Ty(F.getContext()), argTypes_3, false);
        FunctionType *FT_4 = FunctionType::get(Type::getInt64Ty(F.getContext()), argTypes_4, false);

        std::vector<Type*> structTypes(10, Type::getInt8PtrTy(F.getContext()));
        structTypes.insert(structTypes.begin(), Type::getInt32Ty(F.getContext()));
        StructType *ST = StructType::get(F.getContext(), structTypes);

        Function *pFnSSSaveCleanV = Function::Create(FT_4, Function::ExternalLinkage, SLIB_FUNC_SS_SAVE_CLEAN_V, M);
        Function *pFnSSReadV = Function::Create(FT_3, Function::ExternalLinkage, SLIB_FUNC_SS_READ_V, M);
        Function *pFnSSSaveM = Function::Create(FT_3, Function::ExternalLinkage, SLIB_FUNC_SS_SAVE_M, M);
        Function *pFnSSReadM = Function::Create(FT_3, Function::ExternalLinkage, SLIB_FUNC_SS_READ_M, M);
        Function *pFnSSStart = Function::Create(FT_2, Function::ExternalLinkage, SLIB_FUNC_SS_START, M);
        Function *pFnSSExit = Function::Create(FT_2, Function::ExternalLinkage, SLIB_FUNC_SS_EXIT, M);

        GlobalVariable* accessMap = new GlobalVariable(*M, ST, /*isConstant=*/false,
                                          /*Linkage=*/GlobalValue::ExternalLinkage,
                                          /*Initializer=*/0, // has initializer, specified below
                                          /*Name=*/"__channel_access");
        accessMap->setAlignment(8);
        
        Function *fn_ssread = M->getFunction(SLIB_FUNC_SS_READ);
        if (fn_ssread) {
          pFnSSSaveCleanV->copyAttributesFrom(fn_ssread);
          pFnSSReadV->copyAttributesFrom(fn_ssread);
          pFnSSSaveM->copyAttributesFrom(fn_ssread);
          pFnSSReadM->copyAttributesFrom(fn_ssread);
          pFnSSStart->copyAttributesFrom(fn_ssread);
          pFnSSExit->copyAttributesFrom(fn_ssread);
        }
        M->m_pFnSSSaveCleanV = pFnSSSaveCleanV;
        M->m_pFnSSReadV = pFnSSReadV;
        M->m_pFnSSSaveM = pFnSSSaveM;
        M->m_pFnSSReadM = pFnSSReadM;
        M->m_pFnSSStart = pFnSSStart;
        M->m_pFnSSExit = pFnSSExit;
      }

      raw_ostream &O = outs();
      // assumption: @llvm.var.annotation calls are always in the function's entry block.
      std::map<Value *, Value *> mapValueToArgument;
      BasicBlock *b = &F.getEntryBlock();
      
      ymh_log() << "INPUT function\n";
      printLL(F);
      ////////////// <1>
      // run through entry block first to build map of pointers to arguments
      for(BasicBlock::iterator it = b->begin();it!=b->end();++it) {
        Instruction *inst = cast<Instruction> (it);
        if(inst->getOpcode()!=Instruction::Store){
          continue;
        }

        mapValueToArgument[inst->getOperand(1)] = (Value *)inst->getOperand(0);
      }
      printMAP(&mapValueToArgument);

      int idxSSLocalVarsArgs = 0;
      ////////////////// <2>
      // find "llvm.var.annotation" calls
      for (auto it = b->begin(); it != b->end(); it++) {
        if (idxSSLocalVarsArgs >= NO_OF_SS_DATA){
           errs() << "Too many SS_DATA variables \n";
           exit(-1);
        }

        Instruction *I = cast<Instruction> (it);

        if (I->getOpcode() != Instruction::Call) {
          continue;
        }

        Value *calledFunction = I->getOperand(I->getNumOperands()-1);
        if(calledFunction->getName().str() != "llvm.var.annotation")
          continue;

        Value * annotatedValue = I->getOperand(0);
        
        if (mapValueToArgument.count(annotatedValue)) {
          ymh_log() << "HERE?????????????\n";
          annotatedValue = mapValueToArgument[annotatedValue];
        }
        
        Value *a0 = I->getOperand(0);
        Value *annotation = I->getOperand(1);

        Instruction *inst = cast<Instruction> (a0);
        StringRef str1 = getGlobalStringConstant(O, annotation);
        std::string ss = str1.str().substr(0, str1.str().size()-1);

        if (ss_data_annotation.compare(ss)==0) {
          Value *allocaInstr = inst->getOperand(0);
          if (!inSSArgs(allocaInstr)) { // TODO: maybe we don't need this condition
            m_ssLocalVarsArgs[idxSSLocalVarsArgs] = allocaInstr;
            idxSSLocalVarsArgs++;
          }
        }
      }

      ///////////////// <3>
      // add meta data to instruction VAR
    	for (auto& B : F) {
        for (auto& I : B) {
          for(int j=0; j < NO_OF_SS_DATA; j++){
            if (m_ssLocalVarsArgs[j] == &I) {
              addMetadataVAR(I, j);
            } else {
              for(auto op : I.operand_values()) {
                if (m_ssLocalVarsArgs[j] == op) {
                  addMetadataVAR(I, j);
                }
              }
            }
          }

        }
      }

      ///////////////// <4>
      // distinguish local variables and args
      int idxSSArgs = 0;
      for (auto& B : F) {
        for (auto& I : B) {
          if (I.getOpcode() == Instruction::Store && isa<Argument>(I.getOperand(0))) {
          Instruction *allocaInstr = dyn_cast<Instruction>(I.getOperand(1));
          if (allocaInstr && allocaInstr->hasMetadata()) {
            // allocaInstr->print(ymh_log() << "SS_LOCAL_ARG:" << YMH_COLOR_GREEN); errs() << YMH_COLOR_RESET << "\n";
            m_ssArgs[idxSSArgs++] = allocaInstr;
            continue;
          }
        }
        }
      }

      ///////////////// <5>
      // add meta data to instruction ARG
      for (auto& B : F) {
        for (auto& I : B) {
          for(int j=0; j < NO_OF_SS_DATA; j++){
            if (m_ssArgs[j] == &I) {
              addMetadataARG(I, j);
            } else {
              for(auto op : I.operand_values()) 
                if (m_ssArgs[j] == op) addMetadataARG(I, j);
            }
          }
        }
      }

      if (printSSVarArg(F)) printLL(F);

      if (m_ssLocalVarsArgs[0]) F.m_withSSData = Function::WITH_SS_DATA;
      else F.m_withSSData = Function::WITHOUT_SS_DATA;

      ///////////////// <6>
      // add ss_saveCleanV & ss_readV 
      // add ss_saveM & ss_readM
      ymh_log() << "<6> " << F.getName() << "\n";
      if (F.m_withSSData == Function::WITH_SS_DATA) {
        for (auto& B : F) {
          for (auto it = B.begin(); it != B.end(); it++) {
            Instruction *pI = dyn_cast<Instruction>(it);
            if (pI->getOpcode() == Instruction::Call) {
              pI->print(ymh_log() << "Processing INSTR:" << YMH_COLOR_BRIGHT_BLACK);
              errs() << YMH_COLOR_RESET << "\n";
              CallInst *pCallInst = dyn_cast<CallInst>(pI);
              Function *pCallee = pCallInst->getCalledFunction();
              if (pCallee) {
                if (pCallee->getName().str() == "llvm.var.annotation") continue;
                if (isSlibFunc(pCallee) || isLLVMFunc(pCallee)) continue;

                if (isDebugFunc(pCallee)) {
                  ymh_log() << "Found a debug func: " << pCallee->getName() << "()\n";
                  std::vector<Value *> args;

                  unsigned long uuidTop, uuidBottom;
                  if (!getUUID(&uuidTop, &uuidBottom)) {
                    ymh_log() << "CAN'T GET UUID\n";
                    exit(1);
                  }

                  IRBuilder<> IB(pI);
                  args.push_back(IB.getInt64(uuidTop));
                  args.push_back(IB.getInt64(uuidBottom));
                  args.push_back(IB.getInt64(idxSSLocalVarsArgs));
                  ymh_log() << "Creat a call inst for ss_saveM()...\n";
                  Instruction *callToSaveM = CallInst::Create(M->m_pFnSSSaveM, args);
                  ymh_log() << "Inserting ss_saveM()...\n";
                  insertBeforeLastArgLoad(it, B, pCallee, callToSaveM);

                  Instruction *callToReadM = CallInst::Create(M->m_pFnSSReadM, args);
                  B.getInstList().insertAfter(it, callToReadM);

                  continue;
                }
                ymh_log() << F.getName() << "() is calling " << pCallee->getName() << "\n";

                // for all callsite
                IRBuilder<> IB(pI);
                std::vector<Value *> args;
                std::vector<Value *> argLoad;

                ymh_log() << "Inserting anySSLoad\n";
                LoadInst *pAnySSLoad = new LoadInst(m_ssLocalVarsArgs[0], "");
                pAnySSLoad->copyMetadata(*dyn_cast<Instruction>(m_ssLocalVarsArgs[0]));
                insertBeforeLastArgLoad(it, B, pCallee, pAnySSLoad);

                unsigned long uuidTop, uuidBottom;
                if (!getUUID(&uuidTop, &uuidBottom)) {
                  ymh_log() << "CAN'T GET UUID\n";
                  exit(1);
                }

                args.push_back(IB.getInt64(uuidTop));
                args.push_back(IB.getInt64(uuidBottom));
                args.push_back(pAnySSLoad);     // regNo or encoding
                args.push_back(IB.getInt64(0)); // bFuncPtr

                Instruction *callToSaveCleanV = CallInst::Create(M->m_pFnSSSaveCleanV, args);
                insertBeforeLastArgLoad(it, B, pCallee, callToSaveCleanV);
                ymh_log() << "Inserting " << SLIB_FUNC_SS_SAVE_CLEAN_V << "(): " << *callToSaveCleanV << "\n";

                ymh_log() << "Inserting " << SLIB_FUNC_SS_READ_V << "\n";
                Instruction *callToReadM = CallInst::Create(M->m_pFnSSReadV, args);
                B.getInstList().insertAfter(it, callToReadM);

                bModified = true;
              } else {
                // TODO: integrate calling a function pointer with the code above
                Value *pFunctionPointer = pCallInst->getCalledValue();
                if (!pFunctionPointer) {
                  ymh_log() << "UNEXPECTED no function pointer!\n";
                  exit(1);
                } else {
                  ymh_log() << "Function pointer: " << YMH_COLOR_BRIGHT_BLACK << *pFunctionPointer << YMH_COLOR_RESET << "\n";
                  ymh_log() << "Function pointer: " << YMH_COLOR_BRIGHT_BLACK << pFunctionPointer->getName() << YMH_COLOR_RESET << "\n";
                  ymh_log() << "Function pointer: " << YMH_COLOR_BRIGHT_BLACK << *pFunctionPointer->getType() << YMH_COLOR_RESET << "\n";
                }

                IRBuilder<> IB(pI);
                std::vector<Value *> args;
                std::vector<Value *> argLoad;

                ymh_log() << "Inserting anySSLoad\n";
                LoadInst *pAnySSLoad = new LoadInst(m_ssLocalVarsArgs[0], "");
                pAnySSLoad->copyMetadata(*dyn_cast<Instruction>(m_ssLocalVarsArgs[0]));
                insertBeforeLastArgLoad(it, B, NULL, pAnySSLoad);

                unsigned long uuidTop, uuidBottom;
                if (!getUUID(&uuidTop, &uuidBottom)) {
                  ymh_log() << "CAN'T GET UUID\n";
                  exit(1);
                }

                args.push_back(IB.getInt64(uuidTop));
                args.push_back(IB.getInt64(uuidBottom));
                args.push_back(pAnySSLoad);     // regNo or encoding
                args.push_back(IB.getInt64(1)); // bFuncPtr

                Instruction *callToSaveCleanV = CallInst::Create(M->m_pFnSSSaveCleanV, args);
                insertBeforeLastArgLoad(it, B, NULL, callToSaveCleanV);
                ymh_log() << "[CFI_BLR] inserting " << SLIB_FUNC_SS_SAVE_CLEAN_V << "(): " << *callToSaveCleanV << "\n";

                ymh_log() << "Inserting " << SLIB_FUNC_SS_READ_V << "\n";
                Instruction *callToReadM = CallInst::Create(M->m_pFnSSReadV, args);
                B.getInstList().insertAfter(it, callToReadM);

                bModified = true;
              }
            }
          }
        }
      }

      ///////////////// <7>
      // add ss_entry
      /*if (F.m_withSSData == Function::WITH_SS_DATA) {
        BasicBlock &BB = F.getEntryBlock();
        BasicBlock::iterator BBItr = BB.begin();
        std::vector<Value *> args;
        IRBuilder<> IB(&*BBItr);

        unsigned long uuidTop, uuidBottom;
        if (!getUUID(&uuidTop, &uuidBottom)) {
          ymh_log() << "CAN'T GET UUID\n";
          exit(1);
        }

        args.push_back(IB.getInt64(uuidTop));
        args.push_back(IB.getInt64(uuidBottom));
        Instruction *callToEntry = CallInst::Create(M->m_pFnSSStart, args);

        BB.getInstList().insert(BBItr, callToEntry);
      }*/

      if (bModified) ymh_log() << F.getName() << "() is modified\n";
      else ymh_log() << F.getName() << "() is not modified\n";
      return bModified;
    }

    bool getUUID(unsigned long *pTop, unsigned long *pBottom) {
      FILE *fp;
      char uuid[64];
      std::string top, bottom;

      for(int tries = 0; tries < 100; tries++) {
        fp = popen(UUIDGEN_PATH " " UUIDGEN_CMD, "r");
        if (fp == NULL) return false;

        while (fgets(uuid, sizeof(uuid)-1, fp));
        // std::cout << "[" << uuid << "]" << std::endl;
        uuid[strlen(uuid)-1] = '\0';
        // std::cout << "[" << uuid << "] " << strlen(uuid) << std::endl;
        top = uuid;
        bottom = top.substr(19);
        top = top.substr(0,18);

        pclose(fp);

        if (top.length() == 18 && bottom.length() == 18) break;
      }   

      *pTop = std::stoul(top, 0, 16);
      *pBottom = std::stoul(bottom, 0, 16);

      // for debugging....
      /**pTop     = 0x1234567887654321;
      *pBottom  = 0x8765432112345678;*/

      /*std::cout << top << std::endl;
      std::cout << std::hex << std::stoul(top, 0, 16) << " " << top.substr(2) << std::endl;*/
      return true;
    }

    bool quickDecideSSPresence(Function *pFn) {
      for (auto &B : *pFn) {
        for (auto &I : B) {
          if (getSSAnnotation(I)) {
            pFn->m_withSSData = Function::WITH_SS_DATA;
            return true;
          }
        }
      }
      pFn->m_withSSData = Function::WITHOUT_SS_DATA;
      return false;
    }
  };
}
char SkeletonPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerSkeletonPass(const PassManagerBuilder &,
 legacy::PassManagerBase &PM) {
  PM.add(new SkeletonPass());
}
static RegisterStandardPasses
RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
 registerSkeletonPass);
