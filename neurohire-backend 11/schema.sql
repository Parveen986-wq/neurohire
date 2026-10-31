create table if not exists public.analyses (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  filename text,
  candidate_name text,
  ats_score numeric,
  top_role text,
  result jsonb not null
);
grant all on public.analyses to service_role;
alter table public.analyses enable row level security;
