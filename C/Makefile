#####################################################
#   C Source Make File   							#
#####################################################
include ../make.inc
	
kpart_test: eft.h eft.c sum.h sum.c kpart.h kpart.c gentest.h gentest.c kpart_test.c
	$(CC) $(COPT) -o kpart_test eft.c sum.c kpart.c gentest.c kpart_test.c $(CINC) $(CLIB) $(CFLAGS)
	
uninstall:
	@rm -f kpart_test