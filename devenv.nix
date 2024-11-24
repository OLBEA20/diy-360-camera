{ pkgs, lib, config, inputs, ... }:
let 
  pythonPackage = pkgs.python310;
in 
{
  # https://devenv.sh/packages/
  packages = [ pkgs.cmake ];

  enterShell = ''
    export LD_LIBRARY_PATH="${pkgs.libGL}/lib/:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glib.out}/lib:${pkgs.zlib.out}/lib";
  '';

  languages = {
    python = {
      enable = true;
      package = pythonPackage;
      poetry = {
        enable = true;
      };
    };
  };

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
