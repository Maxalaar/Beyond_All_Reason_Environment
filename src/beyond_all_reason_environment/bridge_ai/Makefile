# Build bridge_ai.c into libSkirmishAI.so.
# Usage: make REPO_ROOT=<path-to-repo-root>
# Called automatically by installation.build_bridge_ai().

REPO_ROOT     ?= ../../..
INTERFACE_DIR  = $(REPO_ROOT)/reference/RecoilEngine/rts
CC             = gcc
CFLAGS         = -fPIC -shared -O2 -std=c99 -lm \
                 -DBUILDING_AI -DBUILDING_SKIRMISH_AI \
                 -I$(INTERFACE_DIR)
TARGET         = libSkirmishAI.so

$(TARGET): bridge_ai.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: clean
