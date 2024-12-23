{ pkgs, lib, config, inputs, ... }:
let 
  pythonPackage = pkgs.python310;
in 
{
  # https://devenv.sh/packages/
  packages = [ pkgs.cmake pkgs.hugin ];

  enterShell = ''
    export LD_LIBRARY_PATH="${pkgs.libGL}/lib/:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glib.out}/lib:${pkgs.zlib.out}/lib";
    export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt6.qtbase}/lib/qt-${pkgs.qt6.qtbase.version}/plugins/platforms";
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
