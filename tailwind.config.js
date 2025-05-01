/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./app/**/**/*.{js,ts,jsx,tsx,mdx}", "./app/**/*.{js,ts,jsx,tsx,mdx}", "./pages/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}",

        // Or if using `src` directory:
        "./src/**/*.{js,ts,jsx,tsx,mdx}",], theme: {
        extend: {
            // Optional custom extensions
            // keyframes: {
            //    flip: { /* ... */ }
            // },
            // animation: {
            //    flip: 'flip 0.6s ease-in-out',
            // },
        },
    }, // Dynamically generate grid columns based on constant
    // Note: Tailwind JIT might handle this automatically if used directly in className,
    // but safelisting ensures it's available.
    safelist: [{pattern: /grid-cols-(1|2|3|4|5|6|7|8)/} // Safelist grid columns up to 8 if needed
    ], plugins: [],
}