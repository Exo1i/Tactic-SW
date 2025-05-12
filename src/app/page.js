"use client";
import { useRouter } from "next/navigation";

const games = [
  { name: "Shell Game", id: "shell-game", emoji: "🥚" },
  { name: "Tic Tac Toe", id: "tic-tac-toe", emoji: "❌⭕" },
  { name: "Rubik's Game", id: "rubiks-game", emoji: "🟩🟥🟦" },
  { name: "Memory Matching", id: "memory-matching", emoji: "🧠" },
  { name: "Target Shooter", id: "shooting-game", emoji: "🎯" },
];

export default function Home() {
  const router = useRouter();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-10 bg-gradient-to-br from-blue-100 via-white to-yellow-100">
      <div className="flex flex-col items-center gap-2">
        <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 via-pink-500 to-yellow-500 drop-shadow mb-2">
          🎮 Welcome to the AI Games Hub!
        </h1>
        <p className="text-lg text-gray-700 mb-2 text-center max-w-xl">
          Play with computer vision, robotics, and AI-powered games. Choose a game below and challenge yourself or your friends!
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-2xl">
        {games.map((game, idx) => (
          <button
            key={game.id}
            className={`flex flex-col items-center justify-center px-8 py-8 rounded-2xl shadow-xl bg-white hover:bg-gradient-to-br hover:from-blue-100 hover:to-yellow-100 border-2 border-blue-200 hover:border-yellow-300 transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-400 ${
              idx === games.length - 1 && games.length % 2 === 1
                ? "sm:col-span-2 justify-self-center w-full"
                : ""
            }`}
            onClick={() => router.push(`/games/${game.id}`)}
          >
            <span className="text-5xl mb-2">{game.emoji}</span>
            <span className="text-2xl font-bold text-blue-900 mb-1">{game.name}</span>
            <span className="text-sm text-gray-500">
              {game.id === "shell-game" && "Find the ball under the cup!"}
              {game.id === "tic-tac-toe" && "Classic game, AI never loses!"}
              {game.id === "rubiks-game" && "Watch the robot solve the cube!"}
              {game.id === "memory-matching" && "Test your memory with vision AI!"}
              {game.id === "shooting-game" && "Aim and shoot with computer vision!"}
            </span>
          </button>
        ))}
      </div>
      <footer className="mt-10 text-gray-400 text-xs text-center">
        <span>
          Built with <span className="text-pink-400">♥</span> using Next.js, OpenCV, and AI. <br />
          <span className="text-gray-500">© {new Date().getFullYear()} AI Games Hub</span>
        </span>
      </footer>
    </div>
  );
}
