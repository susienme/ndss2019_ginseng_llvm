//===-- RegAllocFast.cpp - A fast register allocator for debug code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This register allocator allocates registers to a basic block at a time,
// attempting to keep values in registers and reusing registers as appropriate.
//
// Ginseng's RA based on LLVM's Fast register allocator
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include "llvm/MC/MCContext.h"
#include <sstream>
#include "llvm/CodeGen/ginseng.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumStores, "Number of stores added");
STATISTIC(NumLoads , "Number of loads added");
STATISTIC(NumCopies, "Number of copies coalesced");

static RegisterRegAlloc
  fastRegAlloc("ginsengfast", "Ginseng (fast) register allocator", createGinsengFastRegisterAllocator);

namespace {
  class RAGinsengFast : public MachineFunctionPass {
  public:
    static char ID;
    RAGinsengFast() : MachineFunctionPass(ID), StackSlotForVirtReg(-1),
               isBulkSpilling(false) {}

  private:
    MachineFunction *MF;
    MachineRegisterInfo *MRI;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    RegisterClassInfo RegClassInfo;

    // Basic block currently being allocated.
    MachineBasicBlock *MBB;

    // StackSlotForVirtReg - Maps virtual regs to the frame index where these
    // values are spilled.
    IndexedMap<int, VirtReg2IndexFunctor> StackSlotForVirtReg;

    // Everything we know about a live virtual register.
    struct LiveReg {
      MachineInstr *LastUse;    // Last instr to use reg.
      unsigned VirtReg;         // Virtual register number.
      unsigned PhysReg;         // Currently held here.
      unsigned short LastOpNum; // OpNum on LastUse.
      bool Dirty;               // Register needs spill.

      explicit LiveReg(unsigned v)
        : LastUse(nullptr), VirtReg(v), PhysReg(0), LastOpNum(0), Dirty(false){}

      unsigned getSparseSetIndex() const {
        return TargetRegisterInfo::virtReg2Index(VirtReg);
      }
    };

    typedef SparseSet<LiveReg> LiveRegMap;

    // LiveVirtRegs - This map contains entries for each virtual register
    // that is currently available in a physical register.
    LiveRegMap LiveVirtRegs;

    DenseMap<unsigned, SmallVector<MachineInstr *, 4> > LiveDbgValueMap;

    // RegState - Track the state of a physical register.
    enum RegState {
      // A disabled register is not available for allocation, but an alias may
      // be in use. A register can only be moved out of the disabled state if
      // all aliases are disabled.
      regDisabled,

      // A free register is not currently in use and can be allocated
      // immediately without checking aliases.
      regFree,

      // A reserved register has been assigned explicitly (e.g., setting up a
      // call parameter), and it remains reserved until it is used.
      regReserved

      // A register state may also be a virtual register number, indication that
      // the physical register is currently allocated to a virtual register. In
      // that case, LiveVirtRegs contains the inverse mapping.
    };

    // PhysRegState - One of the RegState enums, or a virtreg.
    std::vector<unsigned> PhysRegState;

    // Set of register units.
    typedef SparseSet<unsigned> UsedInInstrSet;

    // Set of register units that are used in the current instruction, and so
    // cannot be allocated.
    UsedInInstrSet UsedInInstr;

    // Mark a physreg as used in this instruction.
    void markRegUsedInInstr(unsigned PhysReg) {
      for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units)
        UsedInInstr.insert(*Units);
    }

    // Check if a physreg or any of its aliases are used in this instruction.
    bool isRegUsedInInstr(unsigned PhysReg) const {
      for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units)
        if (UsedInInstr.count(*Units))
          return true;
      return false;
    }

    // SkippedInstrs - Descriptors of instructions whose clobber list was
    // ignored because all registers were spilled. It is still necessary to
    // mark all the clobbered registers as used by the function.
    SmallPtrSet<const MCInstrDesc*, 4> SkippedInstrs;

    // isBulkSpilling - This flag is set when LiveRegMap will be cleared
    // completely after spilling all live registers. LiveRegMap entries should
    // not be erased.
    bool isBulkSpilling;

    enum : unsigned {
      spillClean = 1,
      spillDirty = 100,
      spillImpossible = ~0u
    };
  public:
    StringRef getPassName() const override { return "Fast Register Allocator"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoPHIs);
    }

    MachineFunctionProperties getSetProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

  private:
    bool runOnMachineFunction(MachineFunction &Fn) override;
    void AllocateBasicBlock();
    void handleThroughOperands(MachineInstr *MI,
                               SmallVectorImpl<unsigned> &VirtDead);
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);
    bool isLastUseOfLocalReg(MachineOperand&);

    void addKillFlag(const LiveReg&);
    void killVirtReg(LiveRegMap::iterator);
    void killVirtReg(unsigned VirtReg);
    void spillVirtReg(MachineBasicBlock::iterator MI, LiveRegMap::iterator);
    void spillVirtReg(MachineBasicBlock::iterator MI, unsigned VirtReg);

    void usePhysReg(MachineOperand&);
    void definePhysReg(MachineInstr &MI, unsigned PhysReg, RegState NewState);
    unsigned calcSpillCost(unsigned PhysReg) const;
    void assignVirtToPhysReg(LiveReg&, unsigned PhysReg);
    LiveRegMap::iterator findLiveVirtReg(unsigned VirtReg) {
      return LiveVirtRegs.find(TargetRegisterInfo::virtReg2Index(VirtReg));
    }
    LiveRegMap::const_iterator findLiveVirtReg(unsigned VirtReg) const {
      return LiveVirtRegs.find(TargetRegisterInfo::virtReg2Index(VirtReg));
    }
    LiveRegMap::iterator assignVirtToPhysReg(unsigned VReg, unsigned PhysReg);
    LiveRegMap::iterator allocVirtReg(MachineInstr &MI, LiveRegMap::iterator,
                                      unsigned Hint);
    LiveRegMap::iterator defineVirtReg(MachineInstr &MI, unsigned OpNum,
                                       unsigned VirtReg, unsigned Hint);
    LiveRegMap::iterator reloadVirtReg(MachineInstr &MI, unsigned OpNum,
                                       unsigned VirtReg, unsigned Hint);
    void spillAll(MachineBasicBlock::iterator MI);
    bool setPhysReg(MachineInstr *MI, unsigned OpNum, unsigned PhysReg);

    unsigned getSSPhyReg(unsigned vreg);
    unsigned getPhyReg(const char *name, const TargetRegisterClass *RC, const TargetRegisterInfo *TRI);

    DenseMap<const Function *, unsigned>                                    *m_pFunc2nrSSVarVRegs = NULL;         // Function -> nr(VREGs)
    DenseMap<const Function *, DenseMap<int, unsigned>*>                    *m_pFunc2VarTag2vreg = NULL;         // Function -> tag2vreg
    DenseMap<const Function *, DenseMap<unsigned, std::vector<unsigned>*>*> *m_pFunc2VarVReg2addedVRegs = NULL;  // Function -> vreg2addedVregs

    DenseMap<const Function *, unsigned>                                    *m_pFunc2nrSSArgVRegs = NULL;         // Function -> nr(VREGs)
    DenseMap<const Function *, DenseMap<int, unsigned>*>                    *m_pFunc2ArgTag2vreg = NULL;         // Function -> tag2vreg
    DenseMap<const Function *, DenseMap<unsigned, std::vector<unsigned>*>*> *m_pFunc2ArgVReg2addedVRegs = NULL;  // Function -> vreg2addedVregs
    DenseMap<const Function *, DenseMap<unsigned, unsigned>*>               *m_pFunc2vreg2argIdx = NULL;        // Function -> vreg2argIdx
    DenseMap<const Function *, DenseMap<unsigned, unsigned>*>               *m_pFunc2phyreg2argIdx = NULL;      // Function -> phyreg2argIdx
    
    DenseMap<int, unsigned>* getVarTag2vreg(const Function *pFn);
    DenseMap<unsigned, std::vector<unsigned>*>* getVarVreg2addedVregs(const Function *pFn);

    DenseMap<int, unsigned>* getArgTag2vreg(const Function *pFn);
    DenseMap<unsigned, std::vector<unsigned>*>* getArgVreg2addedVregs(const Function *pFn);
    DenseMap<unsigned, unsigned>* getVreg2argIdx(const Function *pFn);
    DenseMap<unsigned, unsigned>* getPhyreg2argIdx(const Function *pFn);
    std::vector<unsigned> m_sensitiveFuncPtrPhyRegNo;

    bool isSSReg(MachineFunction *pMFn, unsigned vreg, bool bIncludeAddedVregs = true);
    void printLOG_fromMF(MachineFunction &Fn);
    void printLOG_fromMBB(MachineBasicBlock &MBB);
    void printLOG_fromMI(MachineInstr &MI, bool bLF = true);
    DenseMap<std::vector<unsigned> *, unsigned> m_ssVregs2phyReg;
    int computeSSRegs(MachineFunction &Fn);
    bool contatinPhyReg(unsigned phyReg);
    void cleanSsVregs2phyRegMap();
    void printSsVregs2phyRegMap(MachineFunction &Fn, const TargetRegisterInfo *TRI);
    void printTAGInfo(MachineFunction &Fn);
    void printTAGInfo(MachineFunction &Fn, DenseMap<int, unsigned>* pVarTag2vreg, DenseMap<unsigned, std::vector<unsigned>*>* pVarVreg2addedVregs);
    unsigned vreg2phyReg(unsigned vreg);
    void handleSSReadOpInstrs(MachineFunction &Fn);
    std::vector<std::pair<unsigned, int>> parseVregArgnos(std::string strVregOrgs, const TargetRegisterInfo *TRI);
    bool isCallingSaveCleanV(MachineInstr &MI);
    void checkSensitiveFuncPtr(MachineInstr &MI, MachineBasicBlock &MBB, MachineFunction &Fn);
    bool isSensitiveFuncPtr(unsigned phyRegNo);
    int phyReg2PhyRegNo(unsigned phyReg, const TargetRegisterInfo *TRI);
    int virReg2PhyRegNo(unsigned vreg, const TargetRegisterInfo *TRI);
    int64_t encodeSaveCleanInfo(std::vector<std::pair<unsigned, int>> vregOrgArgnos, MachineFunction &Fn, const TargetRegisterInfo *TRI);
    void printEncodingInfo(int64_t code);
    std::string decode(int eachCode, int regNo);
    enum {
      SAVE_CLEAN_CODE_STATUS_DONTCARE = 0,
      SAVE_CLEAN_CODE_STATUS_CLEAN = 1,
      SAVE_CLEAN_CODE_STATUS_CLEANMOVE = 2,
      SAVE_CLEAN_CODE_STATUS_FUNCPTR = 3
    };
    MachineInstr* findNextCall(MachineInstr *pCurInstr, MachineBasicBlock &MBB);
  };
  char RAGinsengFast::ID = 0;
} // End of Anynomous namespace

INITIALIZE_PASS(RAGinsengFast, "regallocginsengfast", "Ginseng Fast Register Allocator", false, false)

// return tag2Vreg map
// I don't create one because we're in allocator, not selector
DenseMap<int, unsigned>* RAGinsengFast::getVarTag2vreg(const Function *pFn) {
  if (m_pFunc2VarTag2vreg->find(pFn) == m_pFunc2VarTag2vreg->end()) return NULL;    
  return (*m_pFunc2VarTag2vreg)[pFn];
}

// return vreg2vregs map
// I don't create one because we're in allocator, not selector
DenseMap<unsigned, std::vector<unsigned>*>* RAGinsengFast::getVarVreg2addedVregs(const Function *pFn) {
  if (m_pFunc2VarVReg2addedVRegs->find(pFn) == m_pFunc2VarVReg2addedVRegs->end()) {
    ymh_log() << "NULL!\n";
    return NULL;    
  }
  return (*m_pFunc2VarVReg2addedVRegs)[pFn];
}


DenseMap<int, unsigned>* RAGinsengFast::getArgTag2vreg(const Function *pFn) {
  if (m_pFunc2ArgTag2vreg->find(pFn) == m_pFunc2ArgTag2vreg->end()) return NULL;    
  return (*m_pFunc2ArgTag2vreg)[pFn];
}

DenseMap<unsigned, std::vector<unsigned>*>* RAGinsengFast::getArgVreg2addedVregs(const Function *pFn) {
  if (m_pFunc2ArgVReg2addedVRegs->find(pFn) == m_pFunc2ArgVReg2addedVRegs->end()) return NULL;    
  return (*m_pFunc2ArgVReg2addedVRegs)[pFn];
}

DenseMap<unsigned, unsigned>* RAGinsengFast::getVreg2argIdx(const Function *pFn) {
  if (m_pFunc2vreg2argIdx->find(pFn) == m_pFunc2vreg2argIdx->end()) return NULL;    
  return (*m_pFunc2vreg2argIdx)[pFn];
}

// if none, create one.
DenseMap<unsigned, unsigned>* RAGinsengFast::getPhyreg2argIdx(const Function *pFn) {
  if (m_pFunc2phyreg2argIdx->find(pFn) == m_pFunc2phyreg2argIdx->end()) 
    (*m_pFunc2phyreg2argIdx)[pFn] = new DenseMap<unsigned, unsigned>();

  return (*m_pFunc2phyreg2argIdx)[pFn];
}

void RAGinsengFast::printLOG_fromMI(MachineInstr &MI, bool bLF) {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *Fn = MBB->getParent();
  const TargetInstrInfo *TII = Fn->getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = Fn->getSubtarget().getRegisterInfo();

  bool bFound = false;

  ymh_log() << "MINST: "<< TII->getName(MI.getOpcode()) << " ";
  for(unsigned i = 0; i < MI.getNumOperands(); i++) {
    MachineOperand &moperands = MI.getOperand(i);
    MachineOperand::MachineOperandType opType = moperands.getType();

    switch (opType) {
    case MachineOperand::MachineOperandType::MO_Register:
      {
        unsigned reg = moperands.getReg();
        if (  TargetRegisterInfo::isVirtualRegister(reg) ) {
          errs() <<  YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(reg) << YMH_COLOR_RESET << " ";
          if (ginsengIsSSDATA(reg,/* bPrintErr */ false)) bFound = true;
        } else errs() << YMH_COLOR_RED << TRI->getName(reg) << YMH_COLOR_RESET << " ";
      }
      break;

    case MachineOperand::MachineOperandType::MO_FrameIndex:
      errs() << "S_" << moperands.getIndex() << " ";
      break;

    case MachineOperand::MachineOperandType::MO_Immediate:
      errs() << "#" << moperands.getImm() << " ";
      break;

    case MachineOperand::MachineOperandType::MO_GlobalAddress:
      errs() << "gADDR" << " ";
      break;

    case MachineOperand::MachineOperandType::MO_RegisterMask:
      (errs() << "mask(0x").write_hex(*moperands.getRegMask()) << ") ";
      break;

    case MachineOperand::MachineOperandType::MO_Metadata:
      {
        const MDNode *pMDnode = moperands.getMetadata();
        MDString *pMDString = dyn_cast<MDString>(pMDnode->getOperand(0).get());
        if (pMDString) {
          unsigned vreg;
          switch(MI.getOperand(i-1).getImm()) {
          case SLIB_FUNC_SS_READ_IMM:
            vreg = std::stoul(pMDString->getString().str());
            ymh_log() << "TAGGED " 
                      << YMH_COLOR_GREEN << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET
                      << " -> " 
                      << YMH_COLOR_RED << TRI->getName(vreg2phyReg(vreg)) << YMH_COLOR_RESET << " ";
            break;
          case SLIB_FUNC_SS_SAVE_CLEAN_V_IMM:
          default:
            ymh_log() << "TAGGED UNKNOWN: "
                      << *pMDString << "\n";
          }
        }
      }
      break;

    default:
      errs() << "NOT_HANDLED(" << moperands.getType() << ") ";
    }
  }

  if (MI.m_varTagNo != -1) errs() << YMH_COLOR_RED << "<------------ TAGGED "  << MI.m_varTagNo << YMH_COLOR_RESET;

  if (bFound) errs() << YMH_COLOR_RED << "<------------ FOUND"  << YMH_COLOR_RESET;
  if (bLF) errs() << "\n";
}

void RAGinsengFast::printLOG_fromMBB(MachineBasicBlock &MBB) {
  ymh_log() << "Ginseng FAST RA\n";
  for(MachineInstr &minst: MBB) {
    printLOG_fromMI(minst);
  }
}

void RAGinsengFast::printLOG_fromMF(MachineFunction &Fn) {
  ymh_log() << "Ginseng FAST RA\n";

  const Function *pF = Fn.getFunction();
  const Module *pM = pF->getParent();
  unsigned nrRegs;

  if (m_pFunc2nrSSVarVRegs->find(pF) == m_pFunc2nrSSVarVRegs->end()) nrRegs = 0;
  else nrRegs = (*m_pFunc2nrSSVarVRegs)[pF];
  ymh_log() << pF->getName() << "(" << pF << ") in " << pM->getName() << " has " << nrRegs << " tagged regs.\n";
  
  for(MachineBasicBlock &mbb: Fn) {
    printLOG_fromMBB(mbb);
  }
}

/// getStackSpaceFor - This allocates space for the specified virtual register
/// to be held on the stack.
int RAGinsengFast::getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC) {
  // Find the location Reg would belong...
  int SS = StackSlotForVirtReg[VirtReg];
  if (SS != -1)
    return SS;          // Already has space allocated?

  // Allocate a new stack object for this spill location...
  unsigned Size = TRI->getSpillSize(*RC);
  unsigned Align = TRI->getSpillAlignment(*RC);
  int FrameIdx = MF->getFrameInfo().CreateSpillStackObject(Size, Align);

  // Assign the slot.
  StackSlotForVirtReg[VirtReg] = FrameIdx;
  return FrameIdx;
}

/// isLastUseOfLocalReg - Return true if MO is the only remaining reference to
/// its virtual register, and it is guaranteed to be a block-local register.
///
bool RAGinsengFast::isLastUseOfLocalReg(MachineOperand &MO) {
  // If the register has ever been spilled or reloaded, we conservatively assume
  // it is a global register used in multiple blocks.
  if (StackSlotForVirtReg[MO.getReg()] != -1)
    return false;

  // Check that the use/def chain has exactly one operand - MO.
  MachineRegisterInfo::reg_nodbg_iterator I = MRI->reg_nodbg_begin(MO.getReg());
  if (&*I != &MO)
    return false;
  return ++I == MRI->reg_nodbg_end();
}

/// addKillFlag - Set kill flags on last use of a virtual register.
void RAGinsengFast::addKillFlag(const LiveReg &LR) {
  if (!LR.LastUse) return;
  MachineOperand &MO = LR.LastUse->getOperand(LR.LastOpNum);
  if (MO.isUse() && !LR.LastUse->isRegTiedToDefOperand(LR.LastOpNum)) {
    if (MO.getReg() == LR.PhysReg)
      MO.setIsKill();
    // else, don't do anything we are problably redefining a
    // subreg of this register and given we don't track which
    // lanes are actually dead, we cannot insert a kill flag here.
    // Otherwise we may end up in a situation like this:
    // ... = (MO) physreg:sub1, physreg <implicit-use, kill>
    // ... <== Here we would allow later pass to reuse physreg:sub1
    //         which is potentially wrong.
    // LR:sub0 = ...
    // ... = LR.sub1 <== This is going to use physreg:sub1
  }
}

/// killVirtReg - Mark virtreg as no longer available.
void RAGinsengFast::killVirtReg(LiveRegMap::iterator LRI) {
  addKillFlag(*LRI);
  assert(PhysRegState[LRI->PhysReg] == LRI->VirtReg &&
         "Broken RegState mapping");
  PhysRegState[LRI->PhysReg] = regFree;
  // Erase from LiveVirtRegs unless we're spilling in bulk.
  if (!isBulkSpilling)
    LiveVirtRegs.erase(LRI);
}

/// killVirtReg - Mark virtreg as no longer available.
void RAGinsengFast::killVirtReg(unsigned VirtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "killVirtReg needs a virtual register");
  LiveRegMap::iterator LRI = findLiveVirtReg(VirtReg);
  if (LRI != LiveVirtRegs.end())
    killVirtReg(LRI);
}

/// spillVirtReg - This method spills the value specified by VirtReg into the
/// corresponding stack slot if needed.
void RAGinsengFast::spillVirtReg(MachineBasicBlock::iterator MI, unsigned VirtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Spilling a physical register is illegal!");
  LiveRegMap::iterator LRI = findLiveVirtReg(VirtReg);
  assert(LRI != LiveVirtRegs.end() && "Spilling unmapped virtual register");
  spillVirtReg(MI, LRI);
}

/// spillVirtReg - Do the actual work of spilling.
void RAGinsengFast::spillVirtReg(MachineBasicBlock::iterator MI,
                          LiveRegMap::iterator LRI) {
  LiveReg &LR = *LRI;
  assert(PhysRegState[LR.PhysReg] == LRI->VirtReg && "Broken RegState mapping");

  if (LR.Dirty) {
    // If this physreg is used by the instruction, we want to kill it on the
    // instruction, not on the spill.
    bool SpillKill = MachineBasicBlock::iterator(LR.LastUse) != MI;
    LR.Dirty = false;
    DEBUG(dbgs() << "Spilling " << PrintReg(LRI->VirtReg, TRI)
                 << " in " << PrintReg(LR.PhysReg, TRI));
    const TargetRegisterClass *RC = MRI->getRegClass(LRI->VirtReg);
    int FI = getStackSpaceFor(LRI->VirtReg, RC);
    DEBUG(dbgs() << " to stack slot #" << FI << "\n");
    TII->storeRegToStackSlot(*MBB, MI, LR.PhysReg, SpillKill, FI, RC, TRI);
    ++NumStores;   // Update statistics

    // If this register is used by DBG_VALUE then insert new DBG_VALUE to
    // identify spilled location as the place to find corresponding variable's
    // value.
    SmallVectorImpl<MachineInstr *> &LRIDbgValues =
      LiveDbgValueMap[LRI->VirtReg];
    for (unsigned li = 0, le = LRIDbgValues.size(); li != le; ++li) {
      MachineInstr *DBG = LRIDbgValues[li];
      MachineInstr *NewDV = buildDbgValueForSpill(*MBB, MI, *DBG, FI);
      assert(NewDV->getParent() == MBB && "dangling parent pointer");
      (void)NewDV;
      DEBUG(dbgs() << "Inserting debug info due to spill:" << "\n" << *NewDV);
    }
    // Now this register is spilled there is should not be any DBG_VALUE
    // pointing to this register because they are all pointing to spilled value
    // now.
    LRIDbgValues.clear();
    if (SpillKill)
      LR.LastUse = nullptr; // Don't kill register again
  }
  killVirtReg(LRI);
}

/// spillAll - Spill all dirty virtregs without killing them.
void RAGinsengFast::spillAll(MachineBasicBlock::iterator MI) {
  if (LiveVirtRegs.empty()) return;
  isBulkSpilling = true;
  // The LiveRegMap is keyed by an unsigned (the virtreg number), so the order
  // of spilling here is deterministic, if arbitrary.
  for (LiveRegMap::iterator i = LiveVirtRegs.begin(), e = LiveVirtRegs.end();
       i != e; ++i)
    spillVirtReg(MI, i);
  LiveVirtRegs.clear();
  isBulkSpilling = false;
}

/// usePhysReg - Handle the direct use of a physical register.
/// Check that the register is not used by a virtreg.
/// Kill the physreg, marking it free.
/// This may add implicit kills to MO->getParent() and invalidate MO.
void RAGinsengFast::usePhysReg(MachineOperand &MO) {
  unsigned PhysReg = MO.getReg();
  assert(TargetRegisterInfo::isPhysicalRegister(PhysReg) &&
         "Bad usePhysReg operand");

  // Ignore undef uses.
  if (MO.isUndef())
    return;

  markRegUsedInInstr(PhysReg);
  switch (PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  case regReserved:
    PhysRegState[PhysReg] = regFree;
    LLVM_FALLTHROUGH;
  case regFree:
    MO.setIsKill();
    return;
  default:
    // The physreg was allocated to a virtual register. That means the value we
    // wanted has been clobbered.
    llvm_unreachable("Instruction uses an allocated register");
  }

  // Maybe a superregister is reserved?
  for (MCRegAliasIterator AI(PhysReg, TRI, false); AI.isValid(); ++AI) {
    unsigned Alias = *AI;
    switch (PhysRegState[Alias]) {
    case regDisabled:
      break;
    case regReserved:
      // Either PhysReg is a subregister of Alias and we mark the
      // whole register as free, or PhysReg is the superregister of
      // Alias and we mark all the aliases as disabled before freeing
      // PhysReg.
      // In the latter case, since PhysReg was disabled, this means that
      // its value is defined only by physical sub-registers. This check
      // is performed by the assert of the default case in this loop.
      // Note: The value of the superregister may only be partial
      // defined, that is why regDisabled is a valid state for aliases.
      assert((TRI->isSuperRegister(PhysReg, Alias) ||
              TRI->isSuperRegister(Alias, PhysReg)) &&
             "Instruction is not using a subregister of a reserved register");
      LLVM_FALLTHROUGH;
    case regFree:
      if (TRI->isSuperRegister(PhysReg, Alias)) {
        // Leave the superregister in the working set.
        PhysRegState[Alias] = regFree;
        MO.getParent()->addRegisterKilled(Alias, TRI, true);
        return;
      }
      // Some other alias was in the working set - clear it.
      PhysRegState[Alias] = regDisabled;
      break;
    default:
      llvm_unreachable("Instruction uses an alias of an allocated register");
    }
  }

  // All aliases are disabled, bring register into working set.
  PhysRegState[PhysReg] = regFree;
  MO.setIsKill();
}

/// definePhysReg - Mark PhysReg as reserved or free after spilling any
/// virtregs. This is very similar to defineVirtReg except the physreg is
/// reserved instead of allocated.
void RAGinsengFast::definePhysReg(MachineInstr &MI, unsigned PhysReg,
                           RegState NewState) {
  markRegUsedInInstr(PhysReg);
  switch (unsigned VirtReg = PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  default:
    spillVirtReg(MI, VirtReg);
    LLVM_FALLTHROUGH;
  case regFree:
  case regReserved:
    PhysRegState[PhysReg] = NewState;
    return;
  }

  // This is a disabled register, disable all aliases.
  PhysRegState[PhysReg] = NewState;
  for (MCRegAliasIterator AI(PhysReg, TRI, false); AI.isValid(); ++AI) {
    unsigned Alias = *AI;
    switch (unsigned VirtReg = PhysRegState[Alias]) {
    case regDisabled:
      break;
    default:
      spillVirtReg(MI, VirtReg);
      LLVM_FALLTHROUGH;
    case regFree:
    case regReserved:
      PhysRegState[Alias] = regDisabled;
      if (TRI->isSuperRegister(PhysReg, Alias))
        return;
      break;
    }
  }
}


// calcSpillCost - Return the cost of spilling clearing out PhysReg and
// aliases so it is free for allocation.
// Returns 0 when PhysReg is free or disabled with all aliases disabled - it
// can be allocated directly.
// Returns spillImpossible when PhysReg or an alias can't be spilled.
unsigned RAGinsengFast::calcSpillCost(unsigned PhysReg) const {
  if (isRegUsedInInstr(PhysReg)) {
    DEBUG(dbgs() << PrintReg(PhysReg, TRI) << " is already used in instr.\n");
    return spillImpossible;
  }
  switch (unsigned VirtReg = PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  case regFree:
    return 0;
  case regReserved:
    DEBUG(dbgs() << PrintReg(VirtReg, TRI) << " corresponding "
                 << PrintReg(PhysReg, TRI) << " is reserved already.\n");
    return spillImpossible;
  default: {
    LiveRegMap::const_iterator I = findLiveVirtReg(VirtReg);
    assert(I != LiveVirtRegs.end() && "Missing VirtReg entry");
    return I->Dirty ? spillDirty : spillClean;
  }
  }

  // This is a disabled register, add up cost of aliases.
  DEBUG(dbgs() << PrintReg(PhysReg, TRI) << " is disabled.\n");
  unsigned Cost = 0;
  for (MCRegAliasIterator AI(PhysReg, TRI, false); AI.isValid(); ++AI) {
    unsigned Alias = *AI;
    switch (unsigned VirtReg = PhysRegState[Alias]) {
    case regDisabled:
      break;
    case regFree:
      ++Cost;
      break;
    case regReserved:
      return spillImpossible;
    default: {
      LiveRegMap::const_iterator I = findLiveVirtReg(VirtReg);
      assert(I != LiveVirtRegs.end() && "Missing VirtReg entry");
      Cost += I->Dirty ? spillDirty : spillClean;
      break;
    }
    }
  }
  return Cost;
}


/// assignVirtToPhysReg - This method updates local state so that we know
/// that PhysReg is the proper container for VirtReg now.  The physical
/// register must not be used for anything else when this is called.
///
void RAGinsengFast::assignVirtToPhysReg(LiveReg &LR, unsigned PhysReg) {
  DEBUG(dbgs() << "Assigning " << PrintReg(LR.VirtReg, TRI) << " to "
               << PrintReg(PhysReg, TRI) << "\n");
  PhysRegState[PhysReg] = LR.VirtReg;
  assert(!LR.PhysReg && "Already assigned a physreg");
  LR.PhysReg = PhysReg;

  ymh_log() << "Assigning " << YMH_COLOR_RED << TRI->getName(PhysReg) << YMH_COLOR_RESET << " to "
            << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::TargetRegisterInfo::virtReg2Index(LR.VirtReg) << YMH_COLOR_RESET << "\n";
}

RAGinsengFast::LiveRegMap::iterator
RAGinsengFast::assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg) {
  LiveRegMap::iterator LRI = findLiveVirtReg(VirtReg);
  assert(LRI != LiveVirtRegs.end() && "VirtReg disappeared");
  assignVirtToPhysReg(*LRI, PhysReg);
  return LRI;
}

unsigned RAGinsengFast::getPhyReg(const char *name, const TargetRegisterClass *RC, const TargetRegisterInfo *TRI) {
  ArrayRef<MCPhysReg> AO = RegClassInfo.getOrder(RC);

  // First try to find a completely free register.
  for (ArrayRef<MCPhysReg>::iterator I = AO.begin(), E = AO.end(); I != E; ++I){
    unsigned PhysReg = *I;
    if ( !strcmp(TRI->getName(PhysReg), name) ) {
      ymh_log() << "Yep! we found a phy reg " << name <<"\n";
      return PhysReg;
    }
  }

  ymh_log() << "Cannot find a reg named " << name << "\n";
  exit(1);
  return 0;
}

bool RAGinsengFast::isSensitiveFuncPtr(unsigned phyRegNo) {
  for (unsigned regno : m_sensitiveFuncPtrPhyRegNo) {
    if (regno == phyRegNo) return true;
  }

  return false;
}

bool RAGinsengFast::contatinPhyReg(unsigned phyReg) {
  for (auto ssVresgs2phyReg : m_ssVregs2phyReg ) {
    if (ssVresgs2phyReg.second == phyReg) return true;
  }

  return false;
}

void RAGinsengFast::cleanSsVregs2phyRegMap() {
  for (auto ssVresgs2phyReg : m_ssVregs2phyReg) {
    delete ssVresgs2phyReg.first;
    ssVresgs2phyReg.first = NULL;
  }
  m_ssVregs2phyReg.clear();
}

void RAGinsengFast::printSsVregs2phyRegMap(MachineFunction &Fn, const TargetRegisterInfo *TRI) {
  for (auto ssVresgs2phyReg : m_ssVregs2phyReg) {
    ymh_log() << Fn.getName() << "(" << &Fn << ") " << YMH_COLOR_GREEN;
    for (auto ssVreg : *(ssVresgs2phyReg.first)) {
      errs() << "%vreg" << TargetRegisterInfo::virtReg2Index(ssVreg) << " ";
    }

    errs()  << YMH_COLOR_RESET << "-> "
            << YMH_COLOR_RED << TRI->getName(ssVresgs2phyReg.second) << YMH_COLOR_RESET "\n";
  }
}

unsigned RAGinsengFast::getSSPhyReg(unsigned vreg) {
  for (auto ssVresgs2phyReg : m_ssVregs2phyReg) {
    for (auto ssVreg : *(ssVresgs2phyReg.first)) {
      if (ssVreg == vreg) return ssVresgs2phyReg.second;
    }
  }

  assert(false && "Looking for a SS_PREG using non SS_VREG");
  return 0;
}

void RAGinsengFast::printTAGInfo(MachineFunction &Fn, 
                                  DenseMap<int, unsigned>* pVarTag2vreg, 
                                  DenseMap<unsigned, std::vector<unsigned>*>* pVarVreg2addedVregs) {
  ymh_log() << Fn.getName() << " has " << pVarTag2vreg->size() << " tags\n";
  for(auto tag2vreg : *pVarTag2vreg) {
    ymh_log() << "TAG " << tag2vreg.first << " has " 
              << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(tag2vreg.second) << YMH_COLOR_RESET
              << " as a primary vreg\n"
              << "    and ";

    if ( (*pVarVreg2addedVregs)[tag2vreg.second] && (*pVarVreg2addedVregs)[tag2vreg.second]->size()) {
      for(unsigned addedVreg : *(*pVarVreg2addedVregs)[tag2vreg.second]) {
        errs() << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(addedVreg) << YMH_COLOR_RESET << " ";
      }
      errs() << "as added vregs\n";
    } else errs() << "no added vregs\n";

  }
}

void RAGinsengFast::printTAGInfo(MachineFunction &Fn) {
  DenseMap<int, unsigned>* pVarTag2vreg = getVarTag2vreg(Fn.getFunction());
  DenseMap<unsigned, std::vector<unsigned>*>* pVarVreg2addedVregs = getVarVreg2addedVregs(Fn.getFunction());
  if (pVarTag2vreg) {
    printTAGInfo(Fn, pVarTag2vreg, pVarVreg2addedVregs);
  } else ymh_log() << Fn.getName() << " has no VAR tag\n";
  
  pVarTag2vreg = getArgTag2vreg(Fn.getFunction());
  pVarVreg2addedVregs = getArgVreg2addedVregs(Fn.getFunction());
  if (pVarTag2vreg) {
    printTAGInfo(Fn, pVarTag2vreg, pVarVreg2addedVregs);
  } else ymh_log() << Fn.getName() << " has no ARG tag\n";
}

unsigned RAGinsengFast::vreg2phyReg(unsigned vreg) {
  for(auto vregs2phyReg : m_ssVregs2phyReg) {
    for(auto _vreg : *vregs2phyReg.first) {
      if (_vreg == vreg) return vregs2phyReg.second;
    }
  }

  return ((unsigned) -1);
}

// compute m_ssVregs2phyReg for the current function
int RAGinsengFast::computeSSRegs(MachineFunction &Fn) {
  // let's clean m_ssVregs2phyReg first.
  cleanSsVregs2phyRegMap();
  ymh_log() << "m_ssVregs2phyReg for " << Fn.getName() << "() is cleared\n";

  // then, compute for the current function
  const TargetRegisterClass *RC = NULL; // = MRI->getRegClass(VirtReg);
  const TargetRegisterInfo *TRI = Fn.getSubtarget().getRegisterInfo();
  int varRegNo = SS_VAR_PHY_REG_START;
  // int argRegNo = SS_ARG_PHY_REG_START;
  char regName[4];
  int totalSSPhyRegs = 0;

  // for VAR
  if (getVarTag2vreg(Fn.getFunction())) {
    for(auto tag2vreg : *getVarTag2vreg(Fn.getFunction())) {
      unsigned vreg = tag2vreg.second;
      if (!RC) RC = MRI->getRegClass(vreg);

      std::vector<unsigned> *vregs = new std::vector<unsigned>();
      sprintf(regName, "X%d", varRegNo);
      unsigned phyReg = getPhyReg(regName, RC, TRI);

      // add primary vreg
      vregs->push_back(vreg);
      // (*getPhyreg2argIdx(Fn.getFunction()))[phyReg] = (*getVreg2argIdx(Fn.getFunction()))[vreg];   // YEP.. this must not be here...

      // add added vregs
      ymh_log() << "Added vregs for "
                << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET
                << "\n";
      if ((*getVarVreg2addedVregs(Fn.getFunction()))[vreg]) {
        ymh_log() << "in if-statement\n";
        for(unsigned addedVreg : *(*getVarVreg2addedVregs(Fn.getFunction()))[vreg]) {
          vregs->push_back(addedVreg);
        }
      }
      ymh_log() << "Added vregs - done\n";

      if (contatinPhyReg(phyReg)) {
        ymh_log() << "THIS MUST NOT HAPPEN\n";
        return -1;
      }
      ymh_log() << "OK so far...\n";

      ymh_log() << YMH_COLOR_GREEN;
      for(auto vreg : *vregs) {
        errs() << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << " ";
      }
      errs() << YMH_COLOR_RESET << "mapped to " << TRI->getName(phyReg) << "\n";
      m_ssVregs2phyReg[vregs] = phyReg;
      totalSSPhyRegs++;

      if ( std::find(Fn.m_ssVarPhyRegs.begin(), Fn.m_ssVarPhyRegs.end(), phyReg) == Fn.m_ssVarPhyRegs.end())
        Fn.m_ssVarPhyRegs.push_back(phyReg);

      if (varRegNo-- < SS_VAR_PHY_REG_END) return -2;
    }
  }

  // for ARG
  if (getArgTag2vreg(Fn.getFunction())) {
    for(auto tag2vreg : *getArgTag2vreg(Fn.getFunction())) {
      unsigned vreg = tag2vreg.second;
      if (!RC) RC = MRI->getRegClass(vreg);

      std::vector<unsigned> *vregs = new std::vector<unsigned>();
      sprintf(regName, "X%d", varRegNo);
      unsigned phyReg = getPhyReg(regName, RC, TRI);

      // add primary vreg
      vregs->push_back(vreg);
      (*getPhyreg2argIdx(Fn.getFunction()))[phyReg] = (*getVreg2argIdx(Fn.getFunction()))[vreg];

      // add added vregs
      if ((*getArgVreg2addedVregs(Fn.getFunction()))[vreg]) {
        for(unsigned addedVreg : *(*getArgVreg2addedVregs(Fn.getFunction()))[vreg]) {
          vregs->push_back(addedVreg);
        }
      }

      if (contatinPhyReg(phyReg)) return -1;

      ymh_log() << YMH_COLOR_GREEN;
      for(auto vreg : *vregs) {
        errs() << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << " ";
      }
      errs() << YMH_COLOR_RESET << "mapped to " << TRI->getName(phyReg) << "\n";
      m_ssVregs2phyReg[vregs] = phyReg;
      totalSSPhyRegs++;

      if ( std::find(Fn.m_ssArgPhyRegs.begin(), Fn.m_ssArgPhyRegs.end(), phyReg) == Fn.m_ssArgPhyRegs.end())
        Fn.m_ssArgPhyRegs.push_back(phyReg);

      // if (argRegNo++ > SS_VAR_PHY_REG_END) return -2;
      if (varRegNo-- < SS_VAR_PHY_REG_END) return -2;
    }
  }

  for (auto &arg : Fn.getFunction()->args()) {
    arg.print(ymh_log() << "ARG: ");
    errs() << "\n";
  }

  printSsVregs2phyRegMap(Fn, TRI);
  return totalSSPhyRegs;
}

bool RAGinsengFast::isSSReg(MachineFunction *pMFn, unsigned vreg, bool bIncludeAddedVregs) {
  // for VAR
  if (getVarTag2vreg(pMFn->getFunction())) {
    for(auto tag2vreg : *getVarTag2vreg(pMFn->getFunction())) {
      // primary SS_REG
      if (tag2vreg.second == vreg) {
        ymh_log() << YMH_COLOR_BRIGHT_CYAN << "[SS_ALLOC] " << YMH_COLOR_RESET
                  << "Found SS VREG "
                  << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET 
                  << "\n";
        return true;
      }

      // added SS_REG
      for(unsigned addedVreg : *(*getVarVreg2addedVregs(pMFn->getFunction()))[tag2vreg.second] ) {
        if (addedVreg == vreg) {
          ymh_log() << YMH_COLOR_BRIGHT_CYAN << "[SS_ALLOC] " << YMH_COLOR_RESET
                    << "Found SS ADDED_VREG "
                    << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET 
                    << "\n";
          return true;
        }
      }
    }
  }

  // for ARG
  if (getArgTag2vreg(pMFn->getFunction())) {
    for(auto tag2vreg : *getArgTag2vreg(pMFn->getFunction())) {
      // primary SS_REG
      if (tag2vreg.second == vreg) {
        ymh_log() << YMH_COLOR_BRIGHT_CYAN << "[SS_ALLOC] " << YMH_COLOR_RESET
                  << "Found SS VREG "
                  << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET 
                  << "\n";
        return true;
      }

      // added SS_REG
      for(unsigned addedVreg : *(*getArgVreg2addedVregs(pMFn->getFunction()))[tag2vreg.second] ) {
        if (addedVreg == vreg) {
          ymh_log() << YMH_COLOR_BRIGHT_CYAN << "[SS_ALLOC] " << YMH_COLOR_RESET
                    << "Found SS ADDED_VREG "
                    << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET 
                    << "\n";
          return true;
        }
      }
    }
  }



  return false;
}

// THIS IS THE FUNCTION!!!!!!!
/// allocVirtReg - Allocate a physical register for VirtReg.
RAGinsengFast::LiveRegMap::iterator RAGinsengFast::allocVirtReg(MachineInstr &MI,
                                                  LiveRegMap::iterator LRI,
                                                  unsigned Hint) {
  const unsigned VirtReg = LRI->VirtReg;


  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Can only allocate virtual registers");

  const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);

  if ( isSSReg(MI.getParent()->getParent(), VirtReg) ) {
    // ymh_log() << "Yep! we found SS vreg\n";
    unsigned PhysReg = getSSPhyReg(VirtReg);//getPhyReg("X15", RC, MI);
    const TargetRegisterInfo *TRI = MI.getParent()->getParent()->getSubtarget().getRegisterInfo();
    ymh_log() << "SS_PHY_REG: " << PhysReg << " for " 
              << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(VirtReg) << YMH_COLOR_RESET
              << "\n";
    ymh_log() << TRI->getName(PhysReg) << "\n";
    assignVirtToPhysReg(*LRI, PhysReg);

    return LRI;
  }

  // Ignore invalid hints.
  if (Hint && (!TargetRegisterInfo::isPhysicalRegister(Hint) ||
               !RC->contains(Hint) || !MRI->isAllocatable(Hint)))
    Hint = 0;

  if (Hint && contatinPhyReg(Hint)) {
    ymh_log() << "Hint contains SSPhyReg " << YMH_COLOR_RED <<TRI->getName(Hint) << YMH_COLOR_RESET << "\n";
    Hint = 0;
  }

  // Take hint when possible.
  if (Hint) {
    // Ignore the hint if we would have to spill a dirty register.
    unsigned Cost = calcSpillCost(Hint);
    if (Cost < spillDirty) {
      if (Cost)
        definePhysReg(MI, Hint, regFree);
      // definePhysReg may kill virtual registers and modify LiveVirtRegs.
      // That invalidates LRI, so run a new lookup for VirtReg.
      return assignVirtToPhysReg(VirtReg, Hint);
    }
  }

  ArrayRef<MCPhysReg> AO = RegClassInfo.getOrder(RC);

  // First try to find a completely free register.
  for (ArrayRef<MCPhysReg>::iterator I = AO.begin(), E = AO.end(); I != E; ++I){
    unsigned PhysReg = *I;
    if (contatinPhyReg(PhysReg)) {
      ymh_log() << "Don't assign SSPhyReg " << YMH_COLOR_RED << TRI->getName(PhysReg) << YMH_COLOR_RESET << "\n";
      continue;
    }
    if (PhysRegState[PhysReg] == regFree && !isRegUsedInInstr(PhysReg)) {
      assignVirtToPhysReg(*LRI, PhysReg);
      return LRI;
    }
  }

  DEBUG(dbgs() << "Allocating " << PrintReg(VirtReg) << " from "
               << TRI->getRegClassName(RC) << "\n");

  unsigned BestReg = 0, BestCost = spillImpossible;
  for (ArrayRef<MCPhysReg>::iterator I = AO.begin(), E = AO.end(); I != E; ++I){
    unsigned Cost = calcSpillCost(*I);
    DEBUG(dbgs() << "\tRegister: " << PrintReg(*I, TRI) << "\n");
    DEBUG(dbgs() << "\tCost: " << Cost << "\n");
    DEBUG(dbgs() << "\tBestCost: " << BestCost << "\n");

    if (contatinPhyReg(*I)) {
      ymh_log() << "Don't assign SSPhyReg " << YMH_COLOR_RED << TRI->getName(*I) << YMH_COLOR_RESET << "\n";
      continue;
    }

    // Cost is 0 when all aliases are already disabled.
    if (Cost == 0) {
      assignVirtToPhysReg(*LRI, *I);
      return LRI;
    }
    if (Cost < BestCost)
      BestReg = *I, BestCost = Cost;
  }

  if (BestReg) {
    definePhysReg(MI, BestReg, regFree);
    // definePhysReg may kill virtual registers and modify LiveVirtRegs.
    // That invalidates LRI, so run a new lookup for VirtReg.
    return assignVirtToPhysReg(VirtReg, BestReg);
  }

  // Nothing we can do. Report an error and keep going with a bad allocation.
  if (MI.isInlineAsm())
    MI.emitError("inline assembly requires more registers than available");
  else
    MI.emitError("ran out of registers during register allocation");
  definePhysReg(MI, *AO.begin(), regFree);
  return assignVirtToPhysReg(VirtReg, *AO.begin());
}

/// defineVirtReg - Allocate a register for VirtReg and mark it as dirty.
RAGinsengFast::LiveRegMap::iterator RAGinsengFast::defineVirtReg(MachineInstr &MI,
                                                   unsigned OpNum,
                                                   unsigned VirtReg,
                                                   unsigned Hint) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Not a virtual register");
  LiveRegMap::iterator LRI;
  bool New;
  std::tie(LRI, New) = LiveVirtRegs.insert(LiveReg(VirtReg));
  if (New) {
    ymh_log() << "NEW\n";
    // If there is no hint, peek at the only use of this register.
    if ((!Hint || !TargetRegisterInfo::isPhysicalRegister(Hint)) &&
        MRI->hasOneNonDBGUse(VirtReg)) {
      ymh_log() << "NO HINT\n";
      const MachineInstr &UseMI = *MRI->use_instr_nodbg_begin(VirtReg);
      // It's a copy, use the destination register as a hint.
      if (UseMI.isCopyLike())
        Hint = UseMI.getOperand(0).getReg();
    }
    ymh_log() << "Hint(" << Hint <<  ")\n";
    LRI = allocVirtReg(MI, LRI, Hint);
  } else if (LRI->LastUse) {
    // Redefining a live register - kill at the last use, unless it is this
    // instruction defining VirtReg multiple times.
    if (LRI->LastUse != &MI || LRI->LastUse->getOperand(LRI->LastOpNum).isUse())
      addKillFlag(*LRI);
  }
  assert(LRI->PhysReg && "Register not assigned");
  LRI->LastUse = &MI;
  LRI->LastOpNum = OpNum;
  LRI->Dirty = true;
  markRegUsedInInstr(LRI->PhysReg);
  return LRI;
}

/// reloadVirtReg - Make sure VirtReg is available in a physreg and return it.
RAGinsengFast::LiveRegMap::iterator RAGinsengFast::reloadVirtReg(MachineInstr &MI,
                                                   unsigned OpNum,
                                                   unsigned VirtReg,
                                                   unsigned Hint) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Not a virtual register");
  LiveRegMap::iterator LRI;
  bool New;
  std::tie(LRI, New) = LiveVirtRegs.insert(LiveReg(VirtReg));
  MachineOperand &MO = MI.getOperand(OpNum);
  if (New) {
    LRI = allocVirtReg(MI, LRI, Hint);
    const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);
    int FrameIndex = getStackSpaceFor(VirtReg, RC);
    DEBUG(dbgs() << "Reloading " << PrintReg(VirtReg, TRI) << " into "
                 << PrintReg(LRI->PhysReg, TRI) << "\n");
    TII->loadRegFromStackSlot(*MBB, MI, LRI->PhysReg, FrameIndex, RC, TRI);
    ++NumLoads;
  } else if (LRI->Dirty) {
    if (isLastUseOfLocalReg(MO)) {
      DEBUG(dbgs() << "Killing last use: " << MO << "\n");
      if (MO.isUse())
        MO.setIsKill();
      else
        MO.setIsDead();
    } else if (MO.isKill()) {
      DEBUG(dbgs() << "Clearing dubious kill: " << MO << "\n");
      MO.setIsKill(false);
    } else if (MO.isDead()) {
      DEBUG(dbgs() << "Clearing dubious dead: " << MO << "\n");
      MO.setIsDead(false);
    }
  } else if (MO.isKill()) {
    // We must remove kill flags from uses of reloaded registers because the
    // register would be killed immediately, and there might be a second use:
    //   %foo = OR %x<kill>, %x
    // This would cause a second reload of %x into a different register.
    DEBUG(dbgs() << "Clearing clean kill: " << MO << "\n");
    MO.setIsKill(false);
  } else if (MO.isDead()) {
    DEBUG(dbgs() << "Clearing clean dead: " << MO << "\n");
    MO.setIsDead(false);
  }
  assert(LRI->PhysReg && "Register not assigned");
  LRI->LastUse = &MI;
  LRI->LastOpNum = OpNum;
  markRegUsedInInstr(LRI->PhysReg);
  return LRI;
}

// setPhysReg - Change operand OpNum in MI the refer the PhysReg, considering
// subregs. This may invalidate any operand pointers.
// Return true if the operand kills its register.
bool RAGinsengFast::setPhysReg(MachineInstr *MI, unsigned OpNum, unsigned PhysReg) {
  MachineOperand &MO = MI->getOperand(OpNum);
  bool Dead = MO.isDead();
  if (!MO.getSubReg()) {
    MO.setReg(PhysReg);
    return MO.isKill() || Dead;
  }

  // Handle subregister index.
  MO.setReg(PhysReg ? TRI->getSubReg(PhysReg, MO.getSubReg()) : 0);
  MO.setSubReg(0);

  // A kill flag implies killing the full register. Add corresponding super
  // register kill.
  if (MO.isKill()) {
    MI->addRegisterKilled(PhysReg, TRI, true);
    return true;
  }

  // A <def,read-undef> of a sub-register requires an implicit def of the full
  // register.
  if (MO.isDef() && MO.isUndef())
    MI->addRegisterDefined(PhysReg, TRI);

  return Dead;
}

// Handle special instruction operand like early clobbers and tied ops when
// there are additional physreg defines.
void RAGinsengFast::handleThroughOperands(MachineInstr *MI,
                                   SmallVectorImpl<unsigned> &VirtDead) {
  DEBUG(dbgs() << "Scanning for through registers:");
  SmallSet<unsigned, 8> ThroughRegs;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (MO.isEarlyClobber() || MI->isRegTiedToDefOperand(i) ||
        (MO.getSubReg() && MI->readsVirtualRegister(Reg))) {
      if (ThroughRegs.insert(Reg).second)
        DEBUG(dbgs() << ' ' << PrintReg(Reg));
    }
  }

  // If any physreg defines collide with preallocated through registers,
  // we must spill and reallocate.
  DEBUG(dbgs() << "\nChecking for physdef collisions.\n");
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef()) continue;
    unsigned Reg = MO.getReg();
    if (!Reg || !TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
    markRegUsedInInstr(Reg);
    for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
      if (ThroughRegs.count(PhysRegState[*AI]))
        definePhysReg(*MI, *AI, regFree);
    }
  }

  SmallVector<unsigned, 8> PartialDefs;
  DEBUG(dbgs() << "Allocating tied uses.\n");
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg)) continue;
    if (MO.isUse()) {
      unsigned DefIdx = 0;
      if (!MI->isRegTiedToDefOperand(i, &DefIdx)) continue;
      DEBUG(dbgs() << "Operand " << i << "("<< MO << ") is tied to operand "
        << DefIdx << ".\n");
      LiveRegMap::iterator LRI = reloadVirtReg(*MI, i, Reg, 0);
      unsigned PhysReg = LRI->PhysReg;
      setPhysReg(MI, i, PhysReg);
      // Note: we don't update the def operand yet. That would cause the normal
      // def-scan to attempt spilling.
    } else if (MO.getSubReg() && MI->readsVirtualRegister(Reg)) {
      DEBUG(dbgs() << "Partial redefine: " << MO << "\n");
      // Reload the register, but don't assign to the operand just yet.
      // That would confuse the later phys-def processing pass.
      LiveRegMap::iterator LRI = reloadVirtReg(*MI, i, Reg, 0);
      PartialDefs.push_back(LRI->PhysReg);
    }
  }

  DEBUG(dbgs() << "Allocating early clobbers.\n");
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg)) continue;
    if (!MO.isEarlyClobber())
      continue;
    // Note: defineVirtReg may invalidate MO.
    LiveRegMap::iterator LRI = defineVirtReg(*MI, i, Reg, 0);
    unsigned PhysReg = LRI->PhysReg;
    if (setPhysReg(MI, i, PhysReg))
      VirtDead.push_back(Reg);
  }

  // Restore UsedInInstr to a state usable for allocating normal virtual uses.
  UsedInInstr.clear();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || (MO.isDef() && !MO.isEarlyClobber())) continue;
    unsigned Reg = MO.getReg();
    if (!Reg || !TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
    DEBUG(dbgs() << "\tSetting " << PrintReg(Reg, TRI)
                 << " as used in instr\n");
    markRegUsedInInstr(Reg);
  }

  // Also mark PartialDefs as used to avoid reallocation.
  for (unsigned i = 0, e = PartialDefs.size(); i != e; ++i)
    markRegUsedInInstr(PartialDefs[i]);
}

void RAGinsengFast::AllocateBasicBlock() {
  DEBUG(dbgs() << "\nAllocating " << *MBB);

  PhysRegState.assign(TRI->getNumRegs(), regDisabled);
  assert(LiveVirtRegs.empty() && "Mapping not cleared from last block?");

  MachineBasicBlock::iterator MII = MBB->begin();

  // Add live-in registers as live.
  for (const auto &LI : MBB->liveins())
    if (MRI->isAllocatable(LI.PhysReg))
      definePhysReg(*MII, LI.PhysReg, regReserved);

  ymh_log() << "##############1\n";
  printLOG_fromMBB(*MBB);
  errs() << "\n";

  SmallVector<unsigned, 8> VirtDead;
  SmallVector<MachineInstr*, 32> Coalesced;

  // Otherwise, sequentially allocate each instruction in the MBB.
  while (MII != MBB->end()) {
    MachineInstr *MI = &*MII++;
    const MCInstrDesc &MCID = MI->getDesc();
    DEBUG({
        dbgs() << "\n>> " << *MI << "Regs:";
        for (unsigned Reg = 1, E = TRI->getNumRegs(); Reg != E; ++Reg) {
          if (PhysRegState[Reg] == regDisabled) continue;
          dbgs() << " " << TRI->getName(Reg);
          switch(PhysRegState[Reg]) {
          case regFree:
            break;
          case regReserved:
            dbgs() << "*";
            break;
          default: {
            dbgs() << '=' << PrintReg(PhysRegState[Reg]);
            LiveRegMap::iterator I = findLiveVirtReg(PhysRegState[Reg]);
            assert(I != LiveVirtRegs.end() && "Missing VirtReg entry");
            if (I->Dirty)
              dbgs() << "*";
            assert(I->PhysReg == Reg && "Bad inverse map");
            break;
          }
          }
        }
        dbgs() << '\n';
        // Check that LiveVirtRegs is the inverse.
        for (LiveRegMap::iterator i = LiveVirtRegs.begin(),
             e = LiveVirtRegs.end(); i != e; ++i) {
           assert(TargetRegisterInfo::isVirtualRegister(i->VirtReg) &&
                  "Bad map key");
           assert(TargetRegisterInfo::isPhysicalRegister(i->PhysReg) &&
                  "Bad map value");
           assert(PhysRegState[i->PhysReg] == i->VirtReg && "Bad inverse map");
        }
      });

    // Debug values are not allowed to change codegen in any way.
    if (MI->isDebugValue()) {
      bool ScanDbgValue = true;
      while (ScanDbgValue) {
        ScanDbgValue = false;
        for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
          MachineOperand &MO = MI->getOperand(i);
          if (!MO.isReg()) continue;
          unsigned Reg = MO.getReg();
          if (!TargetRegisterInfo::isVirtualRegister(Reg)) continue;
          LiveRegMap::iterator LRI = findLiveVirtReg(Reg);
          if (LRI != LiveVirtRegs.end())
            setPhysReg(MI, i, LRI->PhysReg);
          else {
            int SS = StackSlotForVirtReg[Reg];
            if (SS == -1) {
              // We can't allocate a physreg for a DebugValue, sorry!
              DEBUG(dbgs() << "Unable to allocate vreg used by DBG_VALUE");
              MO.setReg(0);
            }
            else {
              // Modify DBG_VALUE now that the value is in a spill slot.
              bool IsIndirect = MI->isIndirectDebugValue();
              if (IsIndirect)
                assert(MI->getOperand(1).getImm() == 0 &&
                       "DBG_VALUE with nonzero offset");
              const MDNode *Var = MI->getDebugVariable();
              const MDNode *Expr = MI->getDebugExpression();
              DebugLoc DL = MI->getDebugLoc();
              MachineBasicBlock *MBB = MI->getParent();
              assert(
                  cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
                  "Expected inlined-at fields to agree");
              MachineInstr *NewDV = BuildMI(*MBB, MBB->erase(MI), DL,
                                            TII->get(TargetOpcode::DBG_VALUE))
                                        .addFrameIndex(SS)
                                        .addImm(0U)
                                        .addMetadata(Var)
                                        .addMetadata(Expr);
              DEBUG(dbgs() << "Modifying debug info due to spill:"
                           << "\t" << *NewDV);
              // Scan NewDV operands from the beginning.
              MI = NewDV;
              ScanDbgValue = true;
              break;
            }
          }
          LiveDbgValueMap[Reg].push_back(MI);
        }
      }
      // Next instruction.
      continue;
    }

    printLOG_fromMI(*MI, /*bLF*/ false);
    errs() << "-0\n";

    // If this is a copy, we may be able to coalesce.
    unsigned CopySrc = 0, CopyDst = 0, CopySrcSub = 0, CopyDstSub = 0;
    if (MI->isCopy()) {
      CopyDst = MI->getOperand(0).getReg();
      CopySrc = MI->getOperand(1).getReg();
      CopyDstSub = MI->getOperand(0).getSubReg();
      CopySrcSub = MI->getOperand(1).getSubReg();
    }

    // Track registers used by instruction.
    UsedInInstr.clear();

    // First scan.
    // Mark physreg uses and early clobbers as used.
    // Find the end of the virtreg operands
    unsigned VirtOpEnd = 0;
    bool hasTiedOps = false;
    bool hasEarlyClobbers = false;
    bool hasPartialRedefs = false;
    bool hasPhysDefs = false;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      // Make sure MRI knows about registers clobbered by regmasks.
      if (MO.isRegMask()) {
        MRI->addPhysRegsUsedFromRegMask(MO.getRegMask());
        continue;
      }
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (!Reg) continue;
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        VirtOpEnd = i+1;
        if (MO.isUse()) {
          hasTiedOps = hasTiedOps ||
                              MCID.getOperandConstraint(i, MCOI::TIED_TO) != -1;
        } else {
          if (MO.isEarlyClobber())
            hasEarlyClobbers = true;
          if (MO.getSubReg() && MI->readsVirtualRegister(Reg))
            hasPartialRedefs = true;
        }
        continue;
      }
      if (!MRI->isAllocatable(Reg)) continue;
      if (MO.isUse()) {
        usePhysReg(MO);
      } else if (MO.isEarlyClobber()) {
        definePhysReg(*MI, Reg,
                      (MO.isImplicit() || MO.isDead()) ? regFree : regReserved);
        hasEarlyClobbers = true;
      } else
        hasPhysDefs = true;
    }

    printLOG_fromMI(*MI, /*bLF*/ false);
    errs() << "-1\n";

    // The instruction may have virtual register operands that must be allocated
    // the same register at use-time and def-time: early clobbers and tied
    // operands. If there are also physical defs, these registers must avoid
    // both physical defs and uses, making them more constrained than normal
    // operands.
    // Similarly, if there are multiple defs and tied operands, we must make
    // sure the same register is allocated to uses and defs.
    // We didn't detect inline asm tied operands above, so just make this extra
    // pass for all inline asm.
    if (MI->isInlineAsm() || hasEarlyClobbers || hasPartialRedefs ||
        (hasTiedOps && (hasPhysDefs || MCID.getNumDefs() > 1))) {
      handleThroughOperands(MI, VirtDead);
      // Don't attempt coalescing when we have funny stuff going on.
      CopyDst = 0;
      // Pretend we have early clobbers so the use operands get marked below.
      // This is not necessary for the common case of a single tied use.
      hasEarlyClobbers = true;
    }
    printLOG_fromMI(*MI, /*bLF*/ false);
    errs() << "-1.5\n";

    // Second scan.
    // Allocate virtreg uses.
    for (unsigned i = 0; i != VirtOpEnd; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(Reg)) continue;
      if (MO.isUse()) {
        LiveRegMap::iterator LRI = reloadVirtReg(*MI, i, Reg, CopyDst);
        unsigned PhysReg = LRI->PhysReg;
        CopySrc = (CopySrc == Reg || CopySrc == PhysReg) ? PhysReg : 0;
        if (setPhysReg(MI, i, PhysReg))
          killVirtReg(LRI);
      }
    }
    printLOG_fromMI(*MI, /*bLF*/ false);
    errs() << "-2\n";

    // Track registers defined by instruction - early clobbers and tied uses at
    // this point.
    UsedInInstr.clear();
    if (hasEarlyClobbers) {
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg()) continue;
        unsigned Reg = MO.getReg();
        if (!Reg || !TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
        // Look for physreg defs and tied uses.
        if (!MO.isDef() && !MI->isRegTiedToDefOperand(i)) continue;
        markRegUsedInInstr(Reg);
      }
    }

    unsigned DefOpEnd = MI->getNumOperands();
    if (MI->isCall()) {
      // Spill all virtregs before a call. This serves one purpose: If an
      // exception is thrown, the landing pad is going to expect to find
      // registers in their spill slots.
      // Note: although this is appealing to just consider all definitions
      // as call-clobbered, this is not correct because some of those
      // definitions may be used later on and we do not want to reuse
      // those for virtual registers in between.
      DEBUG(dbgs() << "  Spilling remaining registers before call.\n");
      spillAll(MI);

      // The imp-defs are skipped below, but we still need to mark those
      // registers as used by the function.
      SkippedInstrs.insert(&MCID);
    }

    printLOG_fromMI(*MI, /*bLF*/ false);
    errs() << "-2.5\n";
    // Third scan.
    // Allocate defs and collect dead defs.
    for (unsigned i = 0; i != DefOpEnd; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() || !MO.getReg() || MO.isEarlyClobber())
        continue;
      unsigned Reg = MO.getReg();

      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        if (!MRI->isAllocatable(Reg)) continue;
        definePhysReg(*MI, Reg, MO.isDead() ? regFree : regReserved);
        continue;
      }

      printLOG_fromMI(*MI, /*bLF*/ false);
      errs() << " CopySrc(" << CopySrc << ")-2.7\n";
      LiveRegMap::iterator LRI = defineVirtReg(*MI, i, Reg, CopySrc);
      unsigned PhysReg = LRI->PhysReg;
      printLOG_fromMI(*MI, /*bLF*/ false);
      errs() << " PhysReg(" << TRI->getName(PhysReg) << ")-2.8\n";
      if (setPhysReg(MI, i, PhysReg)) {
        VirtDead.push_back(Reg);
        CopyDst = 0; // cancel coalescing;
      } else
        CopyDst = (CopyDst == Reg || CopyDst == PhysReg) ? PhysReg : 0;
    }
    printLOG_fromMI(*MI, /*bLF*/ false);
    errs() << "-3\n";

    // Kill dead defs after the scan to ensure that multiple defs of the same
    // register are allocated identically. We didn't need to do this for uses
    // because we are crerating our own kill flags, and they are always at the
    // last use.
    for (unsigned i = 0, e = VirtDead.size(); i != e; ++i)
      killVirtReg(VirtDead[i]);
    VirtDead.clear();

    if (CopyDst && CopyDst == CopySrc && CopyDstSub == CopySrcSub) {
      DEBUG(dbgs() << "-- coalescing: " << *MI);
      Coalesced.push_back(MI);
    } else {
      DEBUG(dbgs() << "<< " << *MI);
    }
  }

  // Spill all physical registers holding virtual registers now.
  DEBUG(dbgs() << "Spilling live registers at end of block.\n");
  spillAll(MBB->getFirstTerminator());

  // Erase all the coalesced copies. We are delaying it until now because
  // LiveVirtRegs might refer to the instrs.
  for (unsigned i = 0, e = Coalesced.size(); i != e; ++i)
    MBB->erase(Coalesced[i]);
  NumCopies += Coalesced.size();

  DEBUG(MBB->dump());
}

std::vector<std::pair<unsigned, int>> RAGinsengFast::parseVregArgnos(std::string strVregOrgArgnos, const TargetRegisterInfo *TRI) {
  std::vector<std::pair<unsigned, int>> vregOrgArgnos;

  bool first = true;
  std::stringstream _strVreg;
  for(char c: strVregOrgArgnos) {
    if (c != '_') _strVreg << c;
    else {
      if (first) {
        vregOrgArgnos.push_back(std::make_pair(std::stoul(_strVreg.str()), 0));
        ymh_log() << "XCALL: vreg -> argNo : " 
                  << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vregOrgArgnos.back().first) << YMH_COLOR_RESET
                  << "(" << YMH_COLOR_RED << virReg2PhyRegNo(vregOrgArgnos.back().first, TRI) << YMH_COLOR_RESET
                  << ") ";
        
        _strVreg.clear();
        _strVreg.str(std::string());
        first = false;
      } else {
        vregOrgArgnos.back().second = std::stoi(_strVreg.str());
        errs() << " -> X" << vregOrgArgnos.back().second << "\n";
        _strVreg.clear();
        _strVreg.str(std::string());
        first = true;
      }
    }
  }

  if (!_strVreg.str().empty()) {
    vregOrgArgnos.back().second = std::stoul(_strVreg.str());
    errs() << " -> X" << vregOrgArgnos.back().second << "\n";
  }

  return vregOrgArgnos;
}

int64_t RAGinsengFast::encodeSaveCleanInfo(std::vector<std::pair<unsigned, int>> vregOrgArgnos,
                                            MachineFunction &Fn,
                                            const TargetRegisterInfo *TRI) {
  int phyRegNo;
  int argNo = -1;
  int64_t codes[8];
  bzero(codes, sizeof(int64_t) * 8);
  int64_t finalCode = 0;

  ymh_log() << "Starting ENCODing...\n";

  // <1> encode explicit request for move & clean
  for(auto vregArgno: vregOrgArgnos) {
    phyRegNo = atoi(TRI->getName(vreg2phyReg(vregArgno.first))+1);
    // destPhyRegNo = atoi(TRI->getName(vreg2phyReg(vregArgno.second))+1);
    argNo = vregArgno.second;

    if(phyRegNo > 15 || phyRegNo < 9) {
      ymh_log() << "Oh No.. src phy reg(" << TRI->getName(vreg2phyReg(vregArgno.first)) << ") is out of bound\n";
      exit(1);
    }

    if (argNo > 7 || argNo < 0) {
      ymh_log() << "Oh No.. argNo(" << argNo << ") is out of bound\n";
      exit(1);
    }

    codes[phyRegNo-8] = (SAVE_CLEAN_CODE_STATUS_CLEANMOVE << 4) | argNo;
    (ymh_log() << "CODE from explicit: 0x").write_hex(codes[phyRegNo-8]) << " for idx(" << (phyRegNo-8) << ") \n";
  }

  // <2> encode implicit req: clean SS regs
  // this function is called after compute ss phy regs (Fn.m_ssVarPhyRegs & Fn.m_ssArgPhyRegs)
  // so, use the information to identify all ss phy regs
  std::vector<unsigned> allPhyRegs = Fn.m_ssVarPhyRegs;
  allPhyRegs.insert(allPhyRegs.begin(), Fn.m_ssArgPhyRegs.begin(), Fn.m_ssArgPhyRegs.end());
  for(unsigned phyReg: allPhyRegs) {
    int phyRegNo = phyReg2PhyRegNo(phyReg, TRI);
    if (phyRegNo > 15 || phyRegNo < 7) {
      ymh_log() << "Oh No.. phy reg(X" << phyRegNo << ") is out of bound\n";
      exit(1);
    }

    if (isSensitiveFuncPtr(phyRegNo)) {
      ymh_log() << "XCALL_FP X" << phyRegNo << " is a function pointer!\n";
      codes[phyRegNo-8] = (SAVE_CLEAN_CODE_STATUS_FUNCPTR << 4);
    } else if (!codes[phyRegNo-8]) {
      codes[phyRegNo-8] = (SAVE_CLEAN_CODE_STATUS_CLEAN << 4);
    }
  }

  // error check
  if (codes[0]) {
    ymh_log() << "Oh no... code for X7 must be zero, but we have " << codes[0] << "\n";
    exit(1);
  }

  // now aggregate array into a single long
  for(int i = 1; i < 8; i++) {
    if (codes[i]) {
      finalCode |= (codes[i] << (i*8));
      ((ymh_log() << "IDX (" << i << ") has ").write_hex((codes[i] << (i*8))) << " finalCode: 0x").write_hex(finalCode) << "\n";
    }
  }

  return finalCode;
}

std::string RAGinsengFast::decode(int eachCode, int regNo) {
  std::stringstream _msg;

  /*if (regNo != -1) {
    _msg << ""
  }*/
  int status = (eachCode & 0xF0) >> 4;
  int dest = (eachCode & 0xF);
  switch(status) {
  case SAVE_CLEAN_CODE_STATUS_DONTCARE:
    _msg << "ignore X" << regNo;
    if (dest) { ymh_log() << "Invalid destination addr X" << dest << " with non-move status\n"; exit(1); }

    break;
  case SAVE_CLEAN_CODE_STATUS_CLEAN:
    _msg << "clean X" << regNo;
    if (dest) { ymh_log() << "Invalid destination addr X" << dest << " with non-move status\n"; exit(1); }
    break;
  case SAVE_CLEAN_CODE_STATUS_CLEANMOVE:
    _msg << "clean & move X" << regNo << " to X" << dest;
    if (dest > 7) {
      ymh_log() << "Invalid destination addr X" << dest <<"\n";
      exit(1);
    }
    break;
  case SAVE_CLEAN_CODE_STATUS_FUNCPTR:
    _msg << "Func_ptr X" << regNo;
    if (dest) { ymh_log() << "Invalid destination addr X" << dest << " with non-move status\n"; exit(1); }
    break;
  default:
    ymh_log() << "unknown status code: " << status << "\n";
    exit(1);
  }

  return _msg.str();
}

void RAGinsengFast::printEncodingInfo(int64_t code) {
  (ymh_log() << "Decoding 0x").write_hex(code) << "\n";
  for(int i = 0; i < 8; i++) {
    int each = (int) ((code & (0xFFUL << (i*8))) >> (i*8));
    
    if (i == 0) ymh_log() << "RES0: " << each << "\n";
    else {
      /*ymh_log() << "( " << code << " & ( 255 << " << (i*8) << ")) >> " << (i*8) << "\n";
      ymh_log() << "EACH: " << each << "\n";*/
      (ymh_log() << "X" << (i+8) << " [0x").write_hex(each) << "] means "
                << decode(each, i+8) << "\n";
    }
  }
}

int RAGinsengFast::phyReg2PhyRegNo(unsigned phyReg, const TargetRegisterInfo *TRI) {
  return atoi(TRI->getName(phyReg)+1);
}

int RAGinsengFast::virReg2PhyRegNo(unsigned vreg, const TargetRegisterInfo *TRI) {
  return atoi(TRI->getName(vreg2phyReg(vreg))+1);
}

MachineInstr* RAGinsengFast::findNextCall(MachineInstr *pCurInstr, MachineBasicBlock &MBB) {
  bool bFoundStartingInstr = false;
  // bool bPassedSaveCleanV = false;
  int dist = 0;
  MachineInstr *rtn = NULL;
  for(MachineInstr &MI : MBB) {
    if (bFoundStartingInstr) {
      dist++;
      if (MI.isCall()) {
        ymh_log() << "Found IS_CALL: " << " DIST: " << dist << MI << "\n";
        ymh_log() << MI.getOperand(0) << "\n";
        rtn = &MI;
        break;
      }
    } else {
      if (&MI == pCurInstr) {
        ymh_log() << "Found Staring point\n";
        bFoundStartingInstr = true;
      }
    }
  }

  return rtn;
}

bool RAGinsengFast::isCallingSaveCleanV(MachineInstr &MI) {
  if (MI.isCall()) {
    MachineOperand &callee = MI.getOperand(0);
    if (callee.isGlobal()) {
      const GlobalValue *pGV = callee.getGlobal();
      if (pGV->getGlobalIdentifier() == SLIB_FUNC_SS_SAVE_CLEAN_V) {
        return true;
      }
    }
  }

  return false;
}

void RAGinsengFast::checkSensitiveFuncPtr(MachineInstr &MI, MachineBasicBlock &MBB, MachineFunction &Fn) {
  const TargetRegisterInfo *TRI = Fn.getSubtarget().getRegisterInfo();
  MachineInstr *pCall = findNextCall(&MI, MBB);
  ymh_log() << "XCALL: Checking Function Pointer...\n";
  if (pCall) {
    if (isCallingSaveCleanV(*pCall)) {  // ss_saveCleanV()
      pCall = findNextCall(pCall, MBB);
      if (pCall) {                      // XCALL
        ymh_log() << "XCALL_NEXT_CALL: " << pCall->getOperand(0) << "\n";
        MachineOperand &callee = pCall->getOperand(0);
        if (callee.isReg()) {           // BL
          unsigned phyReg = vreg2phyReg(callee.getReg());
          if (phyReg != (unsigned) -1) {
            ymh_log() << "XCALL_FP: " << callee << "\n";
            ymh_log() << "XCALL_FP: " << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(callee.getReg()) <<  YMH_COLOR_RESET 
                      << " is mapped to X" << phyReg2PhyRegNo(phyReg, TRI) << "\n";
            m_sensitiveFuncPtrPhyRegNo.push_back(phyReg2PhyRegNo(phyReg, TRI));
          } else {
            ymh_log() << "Ginseng Error: Delcare your function pointer as sensitive!!\n";
            exit(1);
          }
        } // else ymh_log() << "XCALL_FP: Calling FUNC\n";
      } else {
        ymh_log() << "XCALL_NEXT_CALL: Calling NULL???\n";
        exit(1);
      }
    } else {
      ymh_log() << "XCALL: This must be ss_readV(): Calling " << pCall->getOperand(0) << "\n";
      /* ss_readV() -> do nothing*/
    }
    /*
    
    */
  }
}

void RAGinsengFast::handleSSReadOpInstrs(MachineFunction &Fn) {
  const TargetRegisterInfo *TRI = Fn.getSubtarget().getRegisterInfo();
  m_sensitiveFuncPtrPhyRegNo.clear();

  ymh_log() << "FUNC: " << Fn.getName() << "()\n";
  for(MachineBasicBlock &MBB: Fn) {
    unsigned nrMIs = MBB.size();
    ymh_log() << "MBB has " << nrMIs << " instructions\n";
    for(MachineInstr &MI: MBB) {
      for(unsigned i = 0; i < MI.getNumOperands(); i++) {
        MachineOperand &moperands = MI.getOperand(i);

        if (moperands.getType() == MachineOperand::MachineOperandType::MO_Metadata) {
          MI.print(ymh_log() << "Metadata found - " << YMH_COLOR_BRIGHT_BLACK); errs() << YMH_COLOR_RESET;

          MachineInstr *pInst = &MI;
          (ymh_log() << "0x").write_hex((unsigned long) pInst) << " in " << Fn.getFunction()->getName() << "() op" << i << "\n";
          const MDNode *pMDnode = moperands.getMetadata();
          MDString *pMDString = dyn_cast<MDString>(pMDnode->getOperand(0).get());
          if (pMDString) {
            switch(MI.getOperand(i-1).getImm()) {
            case SLIB_FUNC_SS_READ_IMM: {
              unsigned vreg = std::stoul(pMDString->getString().str());
              ymh_log() << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET << "  -->  " << TRI->getName(vreg2phyReg(vreg)) << "\n";
              MI.getOperand(i-1).setImm(atoi(TRI->getName(vreg2phyReg(vreg))+1));
              break;
            }

            case SLIB_FUNC_SS_SAVE_CLEAN_V_IMM: { 
              // This can be for either ss_saveCleanV() or ss_readV()
              // ss_saveCleanV():
              //    1. skip ss_saveCleanV() and find BLR
              //    2. add the dest reg to m_sensitiveFuncPtrPhyRegNo
              // ss_readV():
              //    Do nothing :)
              std::vector<std::pair<unsigned, int>> vregOrgArgnos = parseVregArgnos(pMDString->getString(), TRI);
              
              checkSensitiveFuncPtr(MI, MBB, Fn);
              
              int64_t code = encodeSaveCleanInfo(vregOrgArgnos, Fn, TRI);
              printEncodingInfo(code);
              MI.getOperand(i-1).setImm(code);
              break;
            }

            default:
              ymh_log() << "Unknown IMM: " << MI.getOperand(i-1).getImm() << " with MD: " << *pMDString << "\n";
              exit(1);
            }
          }
        }
      }
    }
  }
}

/// runOnMachineFunction - Register allocate the whole function
///
bool RAGinsengFast::runOnMachineFunction(MachineFunction &Fn) {
  ymh_log() << "********** FAST REGISTER ALLOCATION **********\n";
  ymh_log() << "********** Function: " << Fn.getName() << '\n';
  MF = &Fn;
  MRI = &MF->getRegInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  TII = MF->getSubtarget().getInstrInfo();

  if (!m_pFunc2nrSSVarVRegs) {
    m_pFunc2nrSSVarVRegs = &MF->getContext().m_func2nrSSVarVRegs;
    m_pFunc2VarTag2vreg = &MF->getContext().m_func2VarTag2vreg;
    m_pFunc2VarVReg2addedVRegs = &MF->getContext().m_func2VarVReg2addedVRegs;

    m_pFunc2nrSSArgVRegs = &MF->getContext().m_func2nrSSArgVRegs;
    m_pFunc2ArgTag2vreg = &MF->getContext().m_func2ArgTag2vreg;
    m_pFunc2ArgVReg2addedVRegs = &MF->getContext().m_func2ArgVReg2addedVRegs;
    m_pFunc2vreg2argIdx = &MF->getContext().m_func2vreg2argIdx;
    m_pFunc2phyreg2argIdx = &MF->getContext().m_func2phyreg2argIdx;
  }

  MRI->freezeReservedRegs(Fn);
  RegClassInfo.runOnMachineFunction(Fn);
  UsedInInstr.clear();
  UsedInInstr.setUniverse(TRI->getNumRegUnits());

  // initialize the virtual->physical register map to have a 'null'
  // mapping for all virtual registers
  StackSlotForVirtReg.resize(MRI->getNumVirtRegs());
  LiveVirtRegs.setUniverse(MRI->getNumVirtRegs());

  int nrReservedRegs = computeSSRegs(Fn);
  ymh_log() << "nrReservedRegs: " << nrReservedRegs << "\n";
  // let me check tags and vregs
  if (getVarTag2vreg(Fn.getFunction())) {
    for(auto tag2vreg : *getVarTag2vreg(Fn.getFunction())) {
      ymh_log() << "TAG " << tag2vreg.first << " has " 
                << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(tag2vreg.second) << YMH_COLOR_RESET
                << " which includes:\n";

      if ((*getVarVreg2addedVregs(Fn.getFunction()))[tag2vreg.second]) {
        for (unsigned vreg : *((*getVarVreg2addedVregs(Fn.getFunction()))[tag2vreg.second])) {
          errs() << "    " << YMH_COLOR_GREEN << "%vreg" << TargetRegisterInfo::virtReg2Index(vreg) << YMH_COLOR_RESET << "\n";
        }
      } else errs() << "    None\n";
    }
  }
  printTAGInfo(Fn);

  handleSSReadOpInstrs(Fn);

  errs() << "\n";
  ymh_log() << "Ginseng FAST RA -- BEFORE AllocateBasicBlock()\n";
  printLOG_fromMF(Fn);
  errs() << "\n";
  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBBi = Fn.begin(), MBBe = Fn.end();
       MBBi != MBBe; ++MBBi) {
    MBB = &*MBBi;
    AllocateBasicBlock();
  }
  errs() << "\n";
  ymh_log() << "Ginseng FAST RA -- AFTER AllocateBasicBlock()\n";
  printLOG_fromMF(Fn);
  errs() << "\n";

  // All machine operands and other references to virtual registers have been
  // replaced. Remove the virtual registers.
  MRI->clearVirtRegs();

  SkippedInstrs.clear();
  StackSlotForVirtReg.clear();
  LiveDbgValueMap.clear();

  ymh_log() << "Ginseng FAST RA -- DONE\n";
  return true;
}

FunctionPass *llvm::createGinsengFastRegisterAllocator() {
  return new RAGinsengFast();
}
