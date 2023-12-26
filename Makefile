PREFIX = /usr/local
LIBDIR = lib
INSTALL = install
SED = sed

all:
ifeq ($(OS),Windows_NT)
	$(SED) 's#@@PREFIX@@#$(shell cygpath -m ${PREFIX})#' tops.pc.in > tops.pc
else
	$(SED) 's#@@PREFIX@@#$(PREFIX)#' tops.pc.in > tops.pc
endif

install: all
	$(INSTALL) -m 0755 -d '$(DESTDIR)$(PREFIX)/include/tops'
	$(INSTALL) -m 0644 include/tops/*.h '$(DESTDIR)$(PREFIX)/include/tops'
	$(INSTALL) -m 0755 -d '$(DESTDIR)$(PREFIX)/$(LIBDIR)/pkgconfig'
	$(INSTALL) -m 0644 tops.pc '$(DESTDIR)$(PREFIX)/$(LIBDIR)/pkgconfig'

uninstall:
	rm -rf '$(DESTDIR)$(PREFIX)/include/tops' '$(DESTDIR)$(PREFIX)/$(LIBDIR)/pkgconfig/tops.pc'

.PHONY: all install uninstall

