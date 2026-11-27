import {
  extractEntities,
  filterEntitySubstrings,
} from "../src/utils/entity_extraction";
import type { ExtractedEntity } from "../src/utils/entity_extraction";

describe("extractEntities", () => {
  it("handles product lists, coordinated names, and identifiers", () => {
    const text =
      "User reported top inbound integration pages: OpenClaw 25,443, " +
      "Claude Code 8,916, Codex 2,573, Dify 656. " +
      "User compared Cartesia and Deepgram. " +
      "The email field for Mem0 lives at person.properties.email. " +
      "The qwen endpoint uses person.properties.email. " +
      "Johnson & Johnson was mentioned. " +
      "Glasses around my window. " +
      "On 2026-05-27 there were 90 days of stats.";

    const entityTexts = new Set(
      extractEntities(text).map((entity) => entity.text),
    );
    const normalized = new Set(
      [...entityTexts].map((entityText) => entityText.toLowerCase()),
    );

    for (const expected of [
      "OpenClaw",
      "Claude Code",
      "Codex",
      "Dify",
      "Cartesia",
      "Deepgram",
      "Mem0",
    ]) {
      expect(entityTexts.has(expected)).toBe(true);
    }
    expect(entityTexts.has("person.properties.email")).toBe(true);
    expect(entityTexts.has("qwen endpoint")).toBe(true);
    expect(entityTexts.has("Johnson & Johnson")).toBe(true);
    expect(entityTexts.has("Johnson")).toBe(false);
    expect(normalized.has("top")).toBe(false);
    expect(normalized.has("glasses")).toBe(false);
    expect(entityTexts.has("Cartesia and Deepgram")).toBe(false);
    expect(entityTexts.has("Claude Code 8,916")).toBe(false);
    for (const rejected of ["8,916", "2,573", "656", "2026-05-27", "90"]) {
      expect(entityTexts.has(rejected)).toBe(false);
    }
  });
});


describe("filterEntitySubstrings", () => {
  it("preserves filter semantics and compiles one pattern per entity", () => {
    const entities: ExtractedEntity[] = [
      { type: "PROPER", text: "Johnson & Johnson" },
      { type: "TOPIC", text: "Johnson" },
      { type: "IDENTIFIER", text: "person.properties.email" },
      { type: "IDENTIFIER", text: "personXpropertiesXemail" },
      { type: "TOPIC", text: "mem" },
      { type: "TOPIC", text: "memory" },
      { type: "PROPER", text: "Mem0" },
      { type: "TOPIC", text: "Mem0 SDK" },
    ];

    const NativeRegExp = global.RegExp;
    let constructions = 0;
    global.RegExp = new Proxy(NativeRegExp, {
      construct(target, args, newTarget) {
        constructions += 1;
        return Reflect.construct(target, args, newTarget);
      },
    }) as RegExpConstructor;

    let filtered: ExtractedEntity[];
    try {
      filtered = filterEntitySubstrings(entities);
    } finally {
      global.RegExp = NativeRegExp;
    }

    expect(filtered.map((entity) => entity.text)).toEqual([
      "Johnson & Johnson",
      "person.properties.email",
      "personXpropertiesXemail",
      "mem",
      "memory",
      "Mem0",
      "Mem0 SDK",
    ]);
    expect(constructions).toBe(entities.length);
  });
});
