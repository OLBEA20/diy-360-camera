{ stdenv, lib, qtbase, wrapQtAppsHook, pkgs }: 
let python =
    let
    packageOverrides = self:
    super: {
      opencv4 = super.opencv4.override {
          enableGtk2 = true;
          gtk2 = pkgs.gtk2;
          enableFfmpeg = true; #here is how to add ffmpeg and other compilation flags
        };
    };
    in
    pkgs.python310.override {inherit packageOverrides; self = python;};

in
stdenv.mkDerivation {
  pname = "myapp";
  version = "1.0";

  buildInputs = [ 
    qtbase 
    (python.buildEnv.override {
      extraLibs = [
	      pkgs.python310Packages.matplotlib
	      pkgs.python310Packages.numpy
	      pkgs.python310Packages.scipy
	      pkgs.python310Packages.pip
        python.pkgs.opencv4
      ];
      ignoreCollisions = true;
    })
  ];
  nativeBuildInputs = [ wrapQtAppsHook ]; 
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.libGL}/lib/:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glib.out}/lib:${pkgs.zlib.out}/lib";
  '';
}
