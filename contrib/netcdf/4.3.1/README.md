# Unidata NetCDF

The Unidata network Common Data Form (netCDF) is an interface for
scientific data access and a freely-distributed software library that
provides an implementation of the interface.  The netCDF library also
defines a machine-independent format for representing scientific data.
Together, the interface, library, and format support the creation,
access, and sharing of scientific data.  The current netCDF software
provides C interfaces for applications and data.  Separate software
distributions available from Unidata provide Java, Fortran, and C++
interfaces.  They have been tested on various common platforms.


NetCDF files are self-describing, network-transparent, directly
accessible, and extendible.  `Self-describing` means that a netCDF file
includes information about the data it contains.  `Network-transparent`
means that a netCDF file is represented in a form that can be accessed
by computers with different ways of storing integers, characters, and
floating-point numbers.  `Direct-access` means that a small subset of a
large dataset may be accessed efficiently, without first reading through
all the preceding data.  `Extendible` means that data can be appended to
a netCDF dataset without copying it or redefining its structure.

NetCDF is useful for supporting access to diverse kinds of scientific
data in heterogeneous networking environments and for writing
application software that does not depend on application-specific
formats.  For information about a variety of analysis and display
packages that have been developed to analyze and display data in
netCDF form, see 

* http://www.unidata.ucar.edu/netcdf/software.html

For more information about netCDF, see the netCDF Web page at

* http://www.unidata.ucar.edu/netcdf/

You can obtain a copy of the latest released version of netCDF software
from

* http://www.unidata.ucar.edu/downloads/netcdf

Copyright and licensing information can be found here, as well as in
the COPYRIGHT file accompanying the software

* http://www.unidata.ucar.edu/software/netcdf/copyright.html

To install this package, please see the file INSTALL in the
distribution, or the (possibly more up-to-date) document:

* http://www.unidata.ucar.edu/netcdf/docs/building.html

The netCDF-3 C and FORTRAN-77 interfaces are documented in man(1)
pages at

* http://www.unidata.ucar.edu/netcdf/docs/netcdf-man-3.html
* http://www.unidata.ucar.edu/netcdf/docs/netcdf-man-3f.html 

User's Guides are also available in several forms from the same
location.

A mailing list, netcdfgroup@unidata.ucar.edu, exists for discussion of
the netCDF interface and announcements about netCDF bugs, fixes, and
enhancements.  For information about how to subscribe, see the URL

* http://www.unidata.ucar.edu/netcdf/mailing-lists.html

We appreciate feedback from users of this package.  Please send
comments, suggestions, and bug reports to
<support-netcdf@unidata.ucar.edu>.  Please identify the version of the
package (file VERSION).
