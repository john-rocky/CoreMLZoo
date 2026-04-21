import Foundation

/// CLIP BPE tokenizer implemented in pure Swift. Ported from the Hub App.
///
/// Loads vocabulary + merge rules from the JSON file that ships alongside
/// each CLIP-based mlpackage. Produces token ID sequences compatible with
/// the converted CLIP text encoders.
final class CLIPTokenizer {

    private let encoder: [String: Int]
    private let bpeRanks: [String: Int]
    private let bosToken: Int
    private let eosToken: Int
    let contextLength: Int

    private var cache: [String: [Int]] = [:]

    init(vocabularyURL: URL) throws {
        let data = try Data(contentsOf: vocabularyURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let encoderDict = json["encoder"] as? [String: Int],
              let mergesList = json["merges"] as? [String] else {
            throw CMZError.inferenceFailed(reason: "invalid CLIP vocab JSON")
        }
        self.encoder = encoderDict

        let merges = mergesList.compactMap { line -> (String, String)? in
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { return nil }
            return (String(parts[0]), String(parts[1]))
        }
        self.bpeRanks = Dictionary(uniqueKeysWithValues:
            merges.enumerated().map { ($0.element.0 + " " + $0.element.1, $0.offset) })

        let bosStr = (json["bos_token"] as? String) ?? "<|startoftext|>"
        let eosStr = (json["eos_token"] as? String) ?? "<|endoftext|>"
        self.bosToken = encoderDict[bosStr] ?? 49406
        self.eosToken = encoderDict[eosStr] ?? 49407
        self.contextLength = (json["context_length"] as? Int) ?? 77
    }

    func tokenize(_ text: String) -> [Int] {
        if let cached = cache[text] { return cached }

        let cleaned = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = cleaned.split(separator: " ").map { String($0) }

        var tokens: [Int] = [bosToken]
        for word in words {
            let encoded = byteEncode(word + "</w>")
            for token in bpe(encoded) {
                if let id = encoder[token] { tokens.append(id) }
            }
        }
        tokens.append(eosToken)

        if tokens.count > contextLength {
            tokens = Array(tokens.prefix(contextLength - 1)) + [eosToken]
        }
        while tokens.count < contextLength { tokens.append(0) }

        cache[text] = tokens
        return tokens
    }

    private func byteEncode(_ text: String) -> [String] {
        text.utf8.map { b -> String in
            let i = Int(b)
            if (33...126).contains(i) || (161...172).contains(i) || (174...255).contains(i) {
                return String(Unicode.Scalar(i)!)
            }
            return String(Unicode.Scalar(256 + i)!)
        }
    }

    private func bpe(_ tokens: [String]) -> [String] {
        if tokens.count <= 1 { return tokens }
        var word = tokens
        while true {
            var bestPair: (String, String)?
            var bestRank = Int.max
            for i in 0..<(word.count - 1) {
                let pair = word[i] + " " + word[i + 1]
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank; bestPair = (word[i], word[i + 1])
                }
            }
            guard let (first, second) = bestPair else { break }
            var new: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    new.append(first + second); i += 2
                } else {
                    new.append(word[i]); i += 1
                }
            }
            word = new
            if word.count == 1 { break }
        }
        return word
    }
}
