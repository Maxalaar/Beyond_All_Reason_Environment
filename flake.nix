{
  description = "Beyond All Reason reinforcement learning environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      # Libraries the graphical engine binary (`spring`) loads; spring-headless needs none of them.
      graphicsLibraries = pkgs.symlinkJoin {
        name = "beyond-all-reason-graphics-libraries";
        paths = with pkgs; [
          SDL2
          openal
          libGL
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
          xorg.libXext
          libxkbcommon
          wayland
        ];
      };
    in {
      # Built on demand by match.py (`nix build .#graphics-libraries`), so graphical matches
      # also work from a plain Python or conda environment, without `nix develop`.
      packages.${system}.graphics-libraries = graphicsLibraries;

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python313
          pkgs.p7zip # extracts the engine archive
          pkgs.git
        ];
        # The engine binaries are prebuilt for generic Linux; on NixOS they run through nix-ld.
        shellHook = ''
          export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          export BEYOND_ALL_REASON_GRAPHICS_LIBRARY_PATH="${graphicsLibraries}/lib"
        '';
      };
    };
}
