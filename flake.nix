{
  description = "Beyond All Reason reinforcement learning environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      # Libraries the graphical engine binary (`spring`) loads; spring-headless needs none of them.
      graphicsLibraries = with pkgs; [
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
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python313
          pkgs.p7zip # extracts the engine archive
          pkgs.git
        ];
        # The engine binaries are prebuilt for generic Linux; on NixOS they run through nix-ld.
        # match.py adds BEYOND_ALL_REASON_GRAPHICS_LIBRARY_PATH to LD_LIBRARY_PATH only for graphical matches;
        # /run/opengl-driver/lib holds the system OpenGL driver (NVIDIA or Mesa).
        shellHook = ''
          export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          export BEYOND_ALL_REASON_GRAPHICS_LIBRARY_PATH="${pkgs.lib.makeLibraryPath graphicsLibraries}:/run/opengl-driver/lib"
        '';
      };
    };
}
