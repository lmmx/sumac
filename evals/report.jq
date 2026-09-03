map({
  model,
  scenarios: (.results | length),
  passed: ([.results[] | select(.passed)] | length),
  pass_rate: (([.results[] | select(.passed)] | length) / (.results | length) * 100),
  total_duration_s,
  checks: (
    [.results[].checks | to_entries[]]
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
