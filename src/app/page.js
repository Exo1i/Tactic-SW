"use client";
import { useRouter } from "next/navigation";

const games = [
  { name: "Shell Game", id: "shell-game" },
  { name: "Tic Tac Toe", id: "tic-tac-toe" },
  { name: "Game 2", id: "game-2" },
  { name: "Game 3", id: "game-3" },
  { name: "Game 4", id: "game-4" },
  { name: "Game 5", id: "game-5" },
];

export default function Home() {
  const router = useRouter();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-8">
      <h1 className="text-3xl font-bold mb-4">Choose a Game</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {games.map((game) => (
          <button
            key={game.id}
            className="px-6 py-4 bg-blue-600 text-white rounded shadow hover:bg-blue-700 transition"
            onClick={() => router.push(`/games/${game.id}`)}
          >
            {game.name}
          </button>
        ))}
      </div>
    </div>
  );
}
