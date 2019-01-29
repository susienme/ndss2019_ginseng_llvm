#ifndef __LLVM_CODEGEN_GINSENG_H
#define __LLVM_CODEGEN_GINSENG_H

#define GINSENG_SSDATA_V1

#define SS_VAR_PHY_REG_START	15
#define NR_MAX_SS_DATA_VAR		7
#define SS_VAR_PHY_REG_END		(SS_VAR_PHY_REG_START - NR_MAX_SS_DATA_VAR + 1)

#define SS_ARG_PHY_REG_START	0
#define NR_MAX_SS_DATA_ARG		7
#define SS_ARG_PHY_REG_END		(SS_ARG_PHY_REG_START - NR_MAX_SS_DATA_ARG + 1)

// IMPORTANT: when adding new API, update AArch64FastISel::isSLibFunc()
#define SLIB_FUNC_SS_READ   		"ss_read"
#define SLIB_FUNC_SS_WRITE   		"ss_write"
#define SLIB_FUNC_SS_SAVE_CLEAN_V 	"ss_saveCleanV"
#define SLIB_FUNC_SS_READ_V       	"ss_readV"
#define SLIB_FUNC_SS_SAVE_M 		"ss_saveM"
#define SLIB_FUNC_SS_READ_M 		"ss_readM"
#define SLIB_FUNC_SS_START 			"ss_start"
#define SLIB_FUNC_SS_EXIT 			"ss_exit"
// IMPORTANT: when adding new API, update AArch64FastISel::isSLibFunc()

#define SLIB_FUNC_SS_SAVE_CLEAN_V_ENCODE_ARGIDX	2
#define SLIB_FUNC_SS_READ_IMM				98
#define SLIB_FUNC_SS_SAVE_CLEAN_V_IMM		99
#endif
