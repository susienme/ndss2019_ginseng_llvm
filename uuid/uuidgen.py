#!/usr/bin/env python
import uuid
import sys, getopt

class CMD:
	RUN_APP 		= 0	#default
	RUN_CALLSITES 	= 1

def getUUID():
	newID = '';
	while True:
		newID = uuid.uuid4().hex
		if len(newID) == 32: break
	return newID

def printAppUUID():
	newID = getUUID()

	print('// Autogen UUID: 0x' + newID)
	print('#define TOKEN_APP_TOP \t\t0x' + newID[:16])
	print('#define TOKEN_APP_BOTTOM \t0x' + newID[16:])

def printCallsiteUUID():
	newID = getUUID()
	print('0x' + newID[:16] + ' 0x' + newID[16:])


def printUsage():
	print('usage: ' + sys.argv[0] + ' [app|call]')
	print('\t app is default')

def parseCmdOpe():
	argv = sys.argv

	for each in argv[1:]:
		if each == '-h':
			printUsage()
			sys.exit(0)
		if each == 'call':
			return CMD.RUN_CALLSITES
	return CMD.RUN_APP


def main():
	cmd = parseCmdOpe()
	if cmd == CMD.RUN_CALLSITES: printCallsiteUUID()
	else: printAppUUID()

main()
