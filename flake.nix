{
  description = "Beyond All Reason reinforcement learning environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python313
          pkgs.p7zip # extracts the engine archive
          pkgs.git
        ];
        # The engine binaries are prebuilt for generic Linux; on NixOS they run through nix-ld.
        shellHook = ''
          export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
        '';
      };
    };
}
