map({
  model,
  prompt_variant,
  scenarios: (.results | length),
  passed: ([.results[] | select(.verdict.passed)] | length),
  pass_rate: (([.results[] | select(.verdict.passed)] | length) / (.results | length) * 100),
  total_duration_s,
  mean_tokens_per_sec,
  checks: (
    [.results[].verdict.checks | to_entries[]]
    | group_by(.key)
    | map({
        check: .[0].key,
        passed: ([.[] | select(.value == true)] | length),
        total: length,
        rate: (([.[] | select(.value == true)] | length) / length * 100)
      })
  )
})
| sort_by(-.pass_rate)
