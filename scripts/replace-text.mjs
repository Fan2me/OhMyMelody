import { createReadStream, createWriteStream } from "node:fs";
import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { Transform } from "node:stream";

function printUsage() {
  console.error(
    [
      "Usage:",
      "  node scripts/replace-text.mjs --input <file> --search <text> --replace <text> --output <file>",
      "  node scripts/replace-text.mjs --input <file> --search <text> --replace <text> --in-place",
      "",
      "Options:",
      "  --input     Source TXT file path",
      "  --search    Literal text to search for",
      "  --replace   Replacement text",
      "  --output    Output file path",
      "  --in-place  Replace in the original file",
      "  --encoding  Text encoding, default utf8",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const options = {
    encoding: "utf8",
    inPlace: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--in-place") {
      options.inPlace = true;
      continue;
    }
    if (!token.startsWith("--")) {
      throw new Error(`unknown argument: ${token}`);
    }
    const key = token.slice(2);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`missing value for --${key}`);
    }
    options[key] = value;
    index += 1;
  }

  return options;
}

function createLiteralReplaceStream(search, replace) {
  const overlap = Math.max(0, search.length - 1);
  let tail = "";

  return new Transform({
    decodeStrings: false,
    transform(chunk, _encoding, callback) {
      try {
        const text = tail + String(chunk);
        const safeLength = Math.max(0, text.length - overlap);
        const head = text.slice(0, safeLength);
        tail = text.slice(safeLength);
        callback(null, head.split(search).join(replace));
      } catch (error) {
        callback(error);
      }
    },
    flush(callback) {
      try {
        callback(null, tail.split(search).join(replace));
      } catch (error) {
        callback(error);
      }
    },
  });
}

async function replaceLargeTextFile({
  input,
  output,
  search,
  replace,
  encoding,
}) {
  await fs.mkdir(dirname(output), { recursive: true });
  await pipeline(
    createReadStream(input, { encoding }),
    createLiteralReplaceStream(search, replace),
    createWriteStream(output, { encoding }),
  );
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const input = options.input ? resolve(String(options.input)) : "";
  const search = String(options.search ?? "");
  const replace = String(options.replace ?? "");
  const encoding = String(options.encoding || "utf8");

  if (!input || !search) {
    throw new Error("--input and --search are required");
  }
  if (options.inPlace && options.output) {
    throw new Error("--output and --in-place cannot be used together");
  }

  const output = options.inPlace
    ? `${input}.tmp-replace`
    : options.output
      ? resolve(String(options.output))
      : "";

  if (!output) {
    throw new Error("either --output or --in-place is required");
  }

  await replaceLargeTextFile({
    input,
    output,
    search,
    replace,
    encoding,
  });

  if (options.inPlace) {
    await fs.rename(output, input);
  }

  console.log(`replace done: ${input} -> ${options.inPlace ? input : output}`);
}

main().catch(async (error) => {
  console.error(error instanceof Error ? error.message : String(error));
  printUsage();
  process.exitCode = 1;
});
