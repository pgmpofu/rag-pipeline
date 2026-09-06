## About the Colossus Project

The Colossus Project is an internal initiative started in 2024 to migrate the
legacy billing system to a new event-driven architecture. The project lead is
Maria Chen, and the target completion date is Q2 2027.

## Key Decisions

The team chose Kafka over RabbitMQ for the event bus because it offers better
long-term log retention, which is needed for the audit-replay requirement from
the finance team.

## Open Risks

The biggest open risk is the dependency on the legacy mainframe's nightly batch
export, which has no documented SLA and has caused three outages in the past
year.
