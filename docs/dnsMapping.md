Copy paste from https://www.quora.com/What-action-does-a-DNS-server-take-if-it-does-not-have-an-entry-for-a-requested-URL


DNS will contain a database of IP addresses to DNS domains or zones.

The top of the worldwide DNS hierarchy are root zone databases of which there are now over 1,000 root domains.

These domains are controlled by an organization known as IANA. The top level domains in question are things like: .com, .org, .net, .edu, .gov, etc etc.

When you type a URL into your browser this is (very roughly & basically) what happens (assuming the URL was www.microsoft.com):

    Local DNS cache is checked to see if it has an entry for “www.microsoft.com”
    If one is found it returns the IP to reach “www.microsoft.com” and your browser then queries that IP using HTTP.
    If one is not found your computer has a default DNS server setup with your ISP. It will forward the request to that nameserver via your default gateway.
    If that name server has the entry then it returns the IP and the same as step 2. above occurs.
    If that DNS doesn't have an entry it forwards it to it's default DNS provider through its gateway and a similar process as before happens. Ultimately our browser needs an IP returned which matches to that domain so it can contact the web server via HTTP.
    All domain roots and sub domains have a SOA record (Start of Authority). This is the name server that has authority for that domain zone. Ultimately the name servers will keep forwarding until they reach an SOA name server for that domain or a root level name server. They will then do as before if they finally find a record with the IP for that domain name (URL). They will also cache that record now so the lookup is much easier next time.
    DNS servers rarely need to goto a SOA server or root name server b/c the DNS system is designed to propagate, share parts of these databases, and/or cache their entries.
    Ultimately if your DNS query fails, meaning no DNS anywhere in the world can find that domain (URL) then you will likely end up getting a “domain not found” or “file not found” or some other type of HTTP 404 error.
